# Simulation component contracts

This guide is for a control or GNC engineer implementing their own plant,
controller, estimator, planner, sensors, and actuator models in GNC Lab. It
defines the boundary between your engineering code and the reusable simulation
kernel.

The short version: you own the physics and algorithm meanings; the kernel owns
timing, RK4 propagation, random-stream assignment, update order, and logging.
Components are ordinary Python objects that satisfy the documented methods.
They do not need to inherit from a GNC Lab base class.

## Contract map

The kernel knows only one domain-neutral numerical quantity: the continuous
plant state `x`, represented as a finite, one-dimensional `numpy.ndarray` of
floating-point values. Every other payload is application-owned data. This
allows a cart-pole force, aircraft control-surface command, motor PWM request,
reference trajectory, and filter state to use types that make sense for the
particular system.

| Signal | Owner | Meaning in the engine |
| --- | --- | --- |
| `state` / `x` | Model author | Integrated truth state: a fixed-length, finite 1-D NumPy vector. |
| `command` / `u_cmd` | Controller or command source | Requested control action before an optional actuator. |
| `actuation` / `u_act` | Actuator | Applied discrete actuator output, including saturation, bias, or sampled noise. |
| `plant_input` / `u_plant` | Input builder | Exact input passed to the model during the next held integration interval. |
| `output` / `y_truth` | Model author | Model-defined truth data exposed for logging and sensing. |
| `measurement` / `z` | Sensor | Timestamped imperfect observation of a truth snapshot. |
| `feedback` | Estimator or engine | An estimator result, or an explicit truth `ModelSnapshot` when no estimator exists. |
| `reference` / `r` | Planner | Desired state, path, mode, or other controller target. |

Use small named dataclasses for non-state data rather than anonymous tuples or
loosely documented dictionaries. For example, an aircraft model could define
`AircraftCommand`, `SurfaceDeflection`, `AircraftInput`, `AircraftTruthOutput`,
`ImuMeasurement`, and `AircraftEstimate`. The engine accepts these payloads
without converting them.

```text
controller or command source -> command -> actuator -> actuation
                                      -> input builder -> plant_input -> model

model truth snapshot -> sensors -> measurements -> estimator -> feedback
                                          \-----------------> planner/controller
```

`Any` in the protocol signatures means “your documented application type,” not
“unstructured data is preferred.” Keep the type and units of every payload
unambiguous in the model or experiment documentation.

## Time model and event order

`SimulationConfig` has a fixed `plant_dt`. The duration
`stop_time - start_time` must be an exact integer multiple of it. Sensors,
planners, and controllers use `PeriodicSchedule(period, phase)`; both values
must align to the plant grid. A phase is relative to simulation start, so
`phase=0.02` with `start_time=5.0` first fires at 5.02 s.

At every plant boundary `t_k`, the engine performs this sequence:

1. It creates an incoming truth `ModelSnapshot` from `x(t_k)` and the input
   held over the immediately preceding interval. The model's `output()` fills
   the snapshot's `output` field.
2. Every sensor due at `t_k` samples that incoming snapshot.
3. If an estimator exists and at least one sensor is due, it gets the new
   measurement dictionary through `update()`.
4. Feedback is selected: `estimator.estimate()` when present, otherwise the
   incoming truth `ModelSnapshot`.
5. A due planner updates the reference; otherwise the prior reference is held.
   Before its first update, the reference is `None`.
6. A due controller updates its command. With no controller, an optional
   command source is evaluated at every plant boundary instead. They cannot be
   configured together.
7. At time zero, and whenever a command changes after that, the actuator maps
   command to actuation. Otherwise the prior actuation is held.
8. The input builder combines actuation with model-specific exogenous effects
   to form the outgoing `plant_input`.
9. The engine records the boundary data. Its recorded `output` is evaluated
   with the outgoing input, which matters for direct-feedthrough models.
10. Except at `stop_time`, RK4 propagates the plant using that plant input held
    constant. A configured estimator then gets
    `propagate(t_k, plant_dt, plant_input)`.

At the final boundary, the engine still records and may perform scheduled
discrete updates, but it does not advance the plant or propagate the estimator.

Two details matter when you implement a component:

- A sensor at `t_k` sees the input that brought the plant to `t_k`, not a new
  command calculated at `t_k`. That prevents a hidden measurement-control
  algebraic loop.
- `output()` can be evaluated more than once per boundary and `derivative()` is
  evaluated four times per RK4 interval. Both methods must be deterministic
  and free of externally visible side effects.

## Model contract

A continuous model implements these three methods:

```python
def default_input(self) -> PlantInput: ...


def derivative(
    self,
    time: float,
    state: np.ndarray,
    plant_input: PlantInput,
) -> np.ndarray: ...


def output(
    self,
    time: float,
    state: np.ndarray,
    plant_input: PlantInput,
) -> TruthOutput: ...
```

`default_input()` supplies the input for a minimal open-loop run and the
starting command when `SimulationPlan.initial_command` is omitted. It should
be the explicit condition you want for an uncontrolled experiment: zero torque,
trimmed thrust, or a documented fixed forcing input. It must be a valid
`PlantInput`; “no input” should be intentional, not accidental `None`.

`derivative()` is the continuous RHS, `xdot = f(t, x, u_plant)`. It must:

- return a finite, one-dimensional vector with exactly the same length as
  `state`;
- accept RK4 intermediate states as valid states, not only logged samples;
- treat `state` and `plant_input` as read-only;
- avoid advancing hidden time, drawing random values, writing files, or
  appending to a history used by the calculation.

The engine passes the same `plant_input` to all four RK4 stages. If a force,
command, or disturbance must change within an interval, represent that
behavior as state or use a smaller `plant_dt`; do not mutate the input inside
`derivative()`.

`output()` creates truth data for sensors and logs. It may expose a subset of
state, derived quantities such as airspeed or acceleration, or direct
feedthrough quantities. It must not mutate arguments or have side effects.

Document every model beside its code: state ordering, units, frames, signs,
valid operating region, default input, and validation cases. The kernel does
not silently impose those engineering choices. See
[engineering conventions](conventions.md) for the project-level checklist.

## Sensor contract

Each physical or logical sensor is represented by a named `SensorBinding`:

```python
SensorBinding(
    name="imu",
    sensor=imu_sensor,
    schedule=PeriodicSchedule(period=0.01),
    initial_condition=imu_initial_condition,
)
```

Names must be unique and should remain stable across A/B runs. The engine gives
each name an independent random stream, so adding an unrelated sensor does not
change `"imu"` noise samples for the same run seed.

A sensor implements:

```python
def reset(self, initial_condition: SensorInitialCondition, rng: Generator) -> None: ...


def sample(self, snapshot: ModelSnapshot, rng: Generator) -> MeasurementValue: ...
```

`reset()` is called exactly once before each run. Use it to overwrite all
internal state: stored bias, previous sample time, quantizer state, and initial
calibration. `initial_condition` is exactly the value supplied to the binding;
it may be `None` when your sensor supports a documented default.

`sample()` receives these fields:

| Field | Meaning |
| --- | --- |
| `snapshot.time` | Current sample time. |
| `snapshot.state` | Truth state at that time. |
| `snapshot.plant_input` | Input held over the preceding interval at this boundary. |
| `snapshot.output` | Model-defined truth output. |

At the initial boundary, there is no preceding interval, so `snapshot.plant_input`
is the model's default input before time-zero command processing.

Return only the sensor's value; the engine wraps it as
`Measurement(sensor_name, time, value)` and stores it under the binding name.
One sensor can return a scalar, vector, or typed packet. Use separate bindings
when instruments have different rates, noise models, or failure modes.

Use only the supplied `rng` for sensor noise and stochastic faults. Do not use
global `numpy.random` functions or create a generator inside `sample()`. The
generator persists for that named sensor through a run, so its sequence is
repeatable from `SimulationMetadata.seed`.

## Estimator contract

An estimator owns an internal belief and publishes the feedback type that its
controller or planner needs:

```python
def reset(self, initial_condition: EstimatorInitialCondition, rng: Generator) -> None: ...


def propagate(self, start_time: float, duration: float, plant_input: PlantInput) -> None: ...


def update(self, time: float, measurements: dict[str, Measurement]) -> None: ...


def estimate(self) -> Estimate: ...
```

The engine does not pass plant truth to `propagate()` or `update()`. An EKF,
complementary filter, observer, or other estimator must carry the model
information it needs in its own constructor or internal state.

`propagate()` runs once after every non-final plant interval, even if no sensors
are configured. It receives the exact outgoing `plant_input` used by the plant
for that interval, so it can use the same held-control assumption as truth.

`update()` runs only when one or more sensors are due. Its dictionary contains
only the new samples at that time, keyed by stable binding names. Different
sensors may arrive at different rates. A missing key means that sensor did not
sample; it does not mean a zero measurement.

`estimate()` is called at every boundary after a possible update. It must
always return a valid current feedback object, including before the first
measurement arrives. Return a read-only or copied value rather than a mutable
object that later propagation changes in place.

When an estimator is configured, planner and controller receive its return
value as `feedback`. When none is configured, they receive a `ModelSnapshot`.
This type difference is deliberate: it prevents a perfect-state-feedback
prototype from silently claiming to exercise an estimator. A controller may
support both cases deliberately, or separate adapters can make the distinction
at the experiment boundary.

## Planner contract

A planner is optional and can provide a scalar setpoint, a trajectory, a mode
schedule, a waypoint target, or any other controller reference:

```python
def reset(self, initial_condition: PlannerInitialCondition, rng: Generator) -> None: ...


def update(self, time: float, feedback: Feedback) -> Reference: ...
```

Wrap it in `TimedComponent(planner, schedule, initial_condition)`. It updates
only when due and its last return value is held as controller `reference`
otherwise. Before the first planner update, `reference` is `None`. Align the
initial planner and controller schedules if the controller requires a reference
at time zero, or make the controller safe with `reference is None`.

The planner's feedback follows the estimator/truth selection above. Use the
supplied RNG only when a deliberately stochastic planning experiment is part of
the documented case.

## Controller contract

A controller maps current reference and feedback to an abstract command:

```python
def reset(self, initial_condition: ControllerInitialCondition, rng: Generator) -> None: ...


def update(self, time: float, reference: Reference | None, feedback: Feedback) -> Command: ...
```

Wrap it in `TimedComponent(controller, schedule, initial_condition)`. The
returned command is held until the next controller update, including across
every RK4 interval. A controller does not receive actuator output or plant
input directly; add a measured or modeled effect to your feedback/reference
type if the algorithm truly needs it.

For an early perfect-state-feedback prototype, feedback is a `ModelSnapshot`;
use its `state` and/or `output` deliberately. For an estimated-feedback
prototype, feedback is your estimator's `Estimate`. Define which type the
controller accepts and test nominal and initialization behavior for it.

Continuous controller dynamics belong in the controller's own internal state,
updated at its discrete sample times. If that state must appear in
post-processing, expose it with `snapshot()` as described below.

## Actuator contract

An actuator is an optional discrete map between controller command and applied
actuation:

```python
def reset(self, initial_condition: ActuatorInitialCondition, rng: Generator) -> None: ...


def apply(self, time: float, command: Command, rng: Generator) -> Actuation: ...
```

Use it for zero-order-held effects such as saturation, rate-independent bias,
quantization, sampled command noise, or a discrete update delay represented in
actuator state. It is called once at time zero and then only when a controller
updates command. With an open-loop command source, it is called every boundary
because the source is evaluated every boundary.

Do not put continuous motor, servo, or hydraulic lag dynamics here. Those need
to be part of plant state and the model RHS so RK4 integrates them correctly.
Use the supplied actuator RNG for sampled bias/noise effects, just as for a
sensor.

With no actuator, actuation is a copy of command. Ideal actuation is therefore
the default, rather than a special controller path.

## Open-loop command-source contract

`CommandSource` is an alternative to a controller for scripted inputs and
plant-only experiments:

```python
def command(self, time: float, feedback: Feedback) -> Command: ...
```

It is evaluated at every plant boundary. It has no separate schedule, `reset()`,
or RNG in version 1. Write a pure time-based function for a step, ramp, sine,
or commanded trajectory. If it needs mutable state, random effects, or a slower
update rate, make that behavior explicit in a controller-like component or
model it as plant state rather than hiding it in a command source.

`SimulationPlan` rejects a plan containing both a controller and a command
source, so command ownership stays unambiguous.

## Input-builder contract

The input builder makes the distinction between abstract actuation and the
input expected by a model explicit:

```python
def build(
    self,
    time: float,
    actuation: Actuation,
    snapshot: ModelSnapshot,
) -> PlantInput: ...
```

If no builder is supplied, `IdentityInputBuilder` passes actuation directly to
the model as `plant_input`. Supply one when a model needs an input object such
as `AircraftInput(control_surfaces, wind, mass_properties)` or when static
disturbances are composed at the simulation boundary.

The builder receives the incoming snapshot before the new plant input applies.
Keep it deterministic and free of mutable hidden state: version 1 supplies no
`reset()` or RNG. Time-varying stochastic environment or continuous disturbance
dynamics should be documented plant behavior or a future component extension,
not an unseeded random draw in `build()`.

## Reset, randomness, and component snapshots

The engine calls `reset()` once for every configured planner, controller,
estimator, actuator, and sensor before a run. A component instance can be
reused in sequential runs only if `reset()` fully restores its initial state.
Do not rely on construction-time state surviving a run.

Each component receives an independent persistent NumPy `Generator` derived
from `SimulationMetadata.seed` and a stable component name:

| Component | Stream name |
| --- | --- |
| Planner | `planner` |
| Controller | `controller` |
| Estimator | `estimator` |
| Actuator | `actuator` |
| Sensor named `gps` | `sensor:gps` |

Changing a sensor name deliberately creates a new noise realization. Adding a
different named component does not advance an existing component's stream.
Record the metadata seed, stable names, configuration, and revision whenever
you compare controller or estimator behavior.

## Parameter sweeps and Monte Carlo

The engine executes one fully specified `SimulationPlan` at a time. A parameter
sweep or Monte Carlo study is an experiment-runner responsibility: create one
plan per trial, vary only the documented quantities, run each plan, then reduce
the collected `SimulationResult` objects into figures or metrics.

Model-parameter uncertainty is distinct from sensor/actuator noise. A model
does not receive an engine RNG, so an experiment runner should draw its
parameter set before constructing each model. Use a named experiment-level
stream, such as `NamedRandomStreams(seed).generator("oscillator-frequency")`,
rather than a global generator. Record the exact sampled parameter value and
trial identifier in `SimulationMetadata.attributes`; the attributes are strings
and should use enough precision to round-trip numerical values.

This separation lets you choose a useful comparison deliberately:

- Hold component seeds fixed while changing a model parameter to make paired
  controller comparisons under the same measurement-noise realization.
- Give each trial a documented distinct seed when both parameter and noise
  uncertainty should vary.
- Store the complete parameter set and summary statistics with the generated
  figure, rather than relying on a chart to describe the experiment.

The oscillator-frequency example follows this pattern. It is a teaching
fixture, not a prescription for the number of trials or a statistical
validation method.

To record optional internal state, expose a zero-argument method:

```python
def snapshot(self) -> SerializableComponentState: ...
```

At each plant boundary, the engine deep-copies its result into
`StepRecord.component_states`. It records planner, controller, estimator,
actuator, and sensor snapshots when configured. Return a small serializable
value: for example, an integrator state, covariance diagonal, mode, saturation
flag, or sensor bias. Do not return a live mutable array or an unbounded debug
history. `snapshot()` is optional; a configured component without it appears
with a `None` snapshot.

## Assembling an experiment

The following shape shows the setup code you write around your own components.
It deliberately omits any model or algorithm implementation.

```python
plan = SimulationPlan(
    model=plant,
    initial_state=x0,
    config=SimulationConfig(start_time=0.0, stop_time=20.0, plant_dt=0.002),
    metadata=SimulationMetadata(
        name="cart-pole state-feedback baseline",
        seed=20261018,
        revision="<git commit>",
        tags=("cart-pole", "state-feedback"),
    ),
    planner=TimedComponent(planner, PeriodicSchedule(0.02), planner_x0),
    controller=TimedComponent(controller, PeriodicSchedule(0.01), controller_x0),
    sensors=(SensorBinding("encoder", encoder, PeriodicSchedule(0.01), encoder_x0),),
    estimator=estimator,
    estimator_initial_condition=estimator_x0,
    actuator=actuator,
    actuator_initial_condition=actuator_x0,
    input_builder=input_builder,
)
result = SimulationEngine().run(plan)
```

For perfect-state-feedback first experiments, omit sensors and estimator. The
controller then receives `ModelSnapshot` truth feedback. For a plant-only
response, omit every optional component; the engine uses `default_input()`. For
an open-loop command, omit controller and pass `command_source=`.

`SimulationResult` is currently in memory. It retains `config`, `metadata`,
resolved initial conditions, and one `StepRecord` per plant boundary. Each
record contains truth state/output; command, actuation, and plant input;
feedback/reference; newly sampled measurements; and optional component states.
Post-processing should read this result rather than mutable component instances
after the run.

## Implementation and test checklist

Before treating an algorithm result as engineering evidence, verify:

- State: dimensions, units, frames, ordering, finite-value behavior, and model
  equilibrium or analytic test cases are documented.
- Model: the RHS is side-effect free, accepts RK4 intermediate states, and
  returns a derivative with exactly the expected shape.
- Timing: each schedule is an integer multiple of `plant_dt`; planner and
  controller initialization ordering is deliberate.
- Signals: command, actuation, plant input, output, measurement, estimate, and
  reference have named fields and documented units.
- Randomness: all noise uses the passed generator; the run seed and stable
  component names are recorded.
- Initialization: every `reset()` field has a defined initial value and a
  repeat-run test.
- Logs: values needed in post-processing are returned from `snapshot()` and
  inspected through `SimulationResult`.
- Validation: use an independent simple case, equilibrium, conservation law,
  analytic solution, or known response before relying on a performance plot.

The existing test-only examples and engine test suite demonstrate kernel
contracts. They do not validate any future user-owned model or algorithm; each
needs its own assumptions and validation evidence.
