# Proposed agnostic imitation-learning experiment

Status: design proposal. The controlled environment, independent learner classes,
misspecification certificates, deferred annotation path, and analysis described
below are not implemented by this document. Existing baseline-default changes
must be verified separately. No experimental result or runtime estimate is claimed.

The execution budget is at most one week, including preparation, pilots, failures,
confirmation, and analysis. The objective is to determine when learner-visitation
data helps a restricted learner relative to expert-visitation data at comparable
data budgets. An FTL-DAgger advantage is a hypothesis to test, not a prerequisite
for selecting tasks or reporting results.

## Paper basis and limits

The reference is Li and Zhang, [Agnostic Interactive Imitation Learning: New Theory
and Practical Algorithms](https://arxiv.org/abs/2312.16860), version 2, July 2024.
[Section 3, Theorem 2](https://arxiv.org/html/2312.16860v2#S3) bounds an
episode-level mixture by

$$
J(\hat\pi)-J(\pi^E)
\leq \mu H\left(
\min_{\pi\in\mathcal B}\frac1N\sum_{n=1}^N F_n(\pi)
+\frac{\operatorname{Reg}(N)}N\right),
\qquad
F_n(\pi)=\mathbb E_{s\sim d_{\pi_n}}
\ell(\pi(s),\pi^E(s)).
$$

Here lower cost is better. The approximation term depends on encountered
distributions. [Section 4, Theorem 3](https://arxiv.org/html/2312.16860v2#S4)
proves sublinear regret for MFTPL-P, assuming an exact classification oracle and
sampling access to a covering distribution. It is not an FTL-DAgger guarantee.

[Section 5.1](https://arxiv.org/html/2312.16860v2#S5.SS1) already gives BC fresh
expert-trajectory examples each round. Experiments use continuous actions,
stochastic expert labels, final policies, 10 training seeds, and 25 evaluation
episodes. Figure 1 uses 80% bootstrap bands. Neural optimization is approximate;
MFTPL-P receives covering data from additional DAgger runs.
[Appendix A.1, Proposition 7](https://arxiv.org/html/2312.16860v2#A1.SS1) shows
that no-regret imitation need not minimize task cost within the learner class.

## Questions and falsifiable mechanism

The proposed mechanism is that misspecification causes unavoidable mistakes,
those mistakes expose recovery states absent from expert trajectories, and
interactive labels teach a representable recovery action. This mechanism can
improve task performance without eliminating approximation error.

Keep three concepts separate:

- **Agnostic learning:** expert inclusion in the learner class is not assumed.
  The positive-misspecification conditions below establish exclusion explicitly.
- **Annotation noise:** repeated calls to an expert at an identical full state
  can give different labels. Use a deterministic expert in the primary study.
- **Distribution shift:** learner and expert visit different states. This can
  occur with a realizable learner and can be absent in a misspecified problem.

Primary hypothesis: at a prespecified retained-label budget, FTL-DAgger improves
task performance over BC-iid in conditions with both demonstrable
misspecification and recoverable distribution shift. Also test no-shift and
realizable controls. Do not assume every agnostic task favors interaction.

## Controlled finite MDP

### State space, dynamics, loss, and learner class

Use horizon H and actions {0, 1}. For 0 < alpha < 1/2, each layer has nominal
states N_0 and N_1 and a recovery state R. Every fresh nominal draw selects
N_z with z distributed as Bernoulli(alpha). Episodes begin with a fresh nominal
draw. Time is part of the full MDP state, so this is an ordinary layered MDP.

The deterministic expert chooses z at N_z and q at R, where q is either 0 or 1.
The learner class consists of exactly four stationary policies identified by
(a_N, a_R). Both nominal states must share action a_N; the recovery action is
a_R. The learner restriction, rather than random expert behavior, creates the
aliasing. Learners also ignore the time coordinate.

- At N_z, a correct action gives a fresh nominal state on the next step.
- At N_z, an incorrect action enters R with probability kappa. Otherwise it
  gives a fresh nominal state.
- At R, action q gives a fresh nominal state. The other action stays at R.
- Stop after exactly H actions. The next-state draw after step H is irrelevant.

Set c(s,a) = 1[a != pi^E(s)] and J(pi) = E[sum_t c(s_t,a_t)]. Thus task cost and
cumulative disagreement coincide in this controlled example. This is a
deliberately simple diagnostic, not evidence that imitation loss and task reward
coincide in classical benchmarks. The expert has cost zero. From every state,
following the expert after one deviating action incurs no further cost, so the
recoverability constant is mu = 1 for this loss.

For the alpha = 0 realizable control, remove N_1 from the admissible state
space. The expert then belongs to the four-policy class on its entire state
domain, including R. Merely assigning N_1 zero visitation probability while
keeping conflicting expert labels there would establish only realizability on
the visited support, which is not the intended control.

Use exact enumeration for empirical 0-1 risk minimization and dynamic
programming for expected cost and occupancy. Fix a public, deterministic
tie-breaking rule shared by all algorithms. Neural training and SGD are
unnecessary for this part.

### Certificate and analytical checks

For alpha > 0, both nominal states have positive initial probability and opposite
expert actions, but every learner takes the same action on them. Consequently
the expert is excluded and minimum disagreement under expert visitation is
exactly alpha. No amount of data removes this floor.

At kappa = 0, recovery is unreachable from the initial distribution for every
policy. The optimum class cost is J_B^* = H alpha and all policies encounter the
same nominal-state law. FTL-DAgger and BC-iid with matching sampling and exact
ERM must have identical distributions of learned policies at equal sample counts.

At kappa = 1, the best class policy takes a_N = 0 and a_R = q. Its exact cost is

$$
J_{\mathcal B}^*(H,\alpha)
=\frac{\alpha H}{1+\alpha}
+\frac{\alpha^2}{(1+\alpha)^2}
\left(1-(-\alpha)^H\right).
$$

To check this formula, let r_t be the probability of R before action t.
Then r_1 = 0, r_{t+1} = alpha(1-r_t), and expected step cost is
alpha(1-r_t). Summing the recurrence gives the expression above. The optimal
recovery action costs zero immediately and avoids an incorrect-action loop;
the majority nominal action minimizes the remaining mismatch probability.
Independently enumerate all four policies to verify optimality at every grid
point. The alpha = 0 formula evaluates to zero.

Distinguish expert-relative cost J(pi)-J(pi^E) from class-relative excess cost
J(pi)-J_B^*. Both are nonnegative here, but only the latter subtracts the known
irreducible floor. A change in kappa also changes J_B^*, so comparing raw costs
across kappa conditions alone does not isolate learning quality.

### Grid, collection, and controls

Initial grid: H in {16, 64}, alpha in {0, 0.02, 0.1, 0.3}, and kappa in {0, 1}.
Balance q in {0, 1}, and verify invariance under a global relabeling of actions.
The q variant where the fixed unseen-state prediction is correct is necessary:
do not report only the variant that penalizes BC's arbitrary tie-breaking.

Obtain each theoretical occupancy sample by rolling out an independent episode
and choosing a time uniformly from {1,...,H}. Only the selected state is added
to the training set. For BC-iid, use the expert as behavior; for FTL-DAgger, use
the current learner. This gives iid samples within a fixed-policy batch. Log
the transitions required to obtain them separately. A faster exact-occupancy
sampler can be a validated implementation option, but its resource counts must
not be presented as trajectory collection costs.

Use at least 100 independent training seeds if the pilot confirms this is
inexpensive. Exact evaluation eliminates evaluation-rollout variance, but not
training-data variance. Pair seed blocks and report all q conditions, including
those where BC already recovers successfully.

Required falsifiers and checks:

1. At kappa = 0, a persistent equal-sample advantage indicates a sampling,
   tie-breaking, or implementation discrepancy.
2. At alpha = 0, excess error should be attributable to finite data or training,
   not an asserted positive approximation floor.
3. If improvements depend solely on one action-label orientation, investigate
   the unsupported-state prior before making a mechanism claim.
4. At alpha > 0, claims of convergence to zero expert-relative cost contradict
   the exact floor and require an accounting or environment check.
5. Compute population F_n and Reg(N) exactly on the generated policy sequence.
   Do not infer low regret from an attractive final-return curve.

## Classical benchmark extension

Pilot candidates are [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/),
[Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/),
and [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/).
Select two confirmatory environments using prespecified expert-quality,
misspecification, numerical-stability, and runtime gates. Do not select them by
the sign of the FTL-DAgger versus BC-iid result. CartPole is a useful inexpensive
candidate; it may nevertheless be nearly realizable by a simple linear policy.
MountainCar's useful control rules can also be simple. Freeze the selected
environments and restrictions before confirmatory seeds are run.

Use an independently verified full-observation expert. Freeze its checkpoint,
normalization, deterministic action rule, environment version, and termination
semantics. Expert quality must be measured using fresh evaluation episodes;
being labeled an expert checkpoint is insufficient.

For CartPole, a concrete aliasing pilot bins cart position and pole angle
while coarsening or omitting velocities. For Acrobot, bin the angle features
and coarsen angular velocities. Compare mild and severe coarsening using
bins frozen on independent pilot data. Retain enough information for recovery
to be representable; a restriction that destroys every useful recovery action
can legitimately eliminate any benefit of DAgger. Certify label conflicts in
positive-probability bins before calling a condition agnostic.

Candidate learner classes, in increasing implementation complexity:

- A fixed quantized or aliased observation map with exact majority-per-cell
  fitting. Contradictory deterministic expert labels in the same cell provide
  an explicit representation obstruction.
- A raw-observation linear action-score model, independent of expert features.
- A small independent MLP as a secondary capacity condition if time permits.

The current teacher-feature mode is a realizable construction for the
deterministic teacher action rule when the trainable head can copy the teacher
head. A raw linear head on a complete discrete-state one-hot representation is
also capable of expressing every deterministic tabular policy. Neither should
be described as agnostic merely because the trainable part is linear. Likewise,
a fresh network with the expert's architecture does not establish expert
exclusion. Smaller raw-input models are plausible restrictions, but model size
alone is not a certificate.

For an explicit restricted observation map phi, define a frozen independent
reference distribution d_ref and

$$
\epsilon_{\rm alias}(d_{\rm ref})
=\mathbb E_o\left[1-\max_a
\Pr(\pi^E(s)=a\mid\phi(s)=o)\right].
$$

This lower-bounds the reference-distribution disagreement of every policy using
only phi(s). In the toy, this quantity is known exactly. On a classical task,
opposite-label regions of positive reference probability can certify a positive
floor analytically if their probabilities are established. Otherwise estimate
the finite-cell floor using independent data and simultaneous multinomial or
finite-class confidence bounds, including uncertainty from selecting the
majority label. Report a statistical lower bound only if it is positive.
Training failure or an empirical error alone is not a population certificate.

Choose d_ref and any bins before confirmatory training. A mixture of fresh
expert-visitation states and fixed exploratory states is one option; identify
the mixture weights and collection method. Its diagnostic annotations cannot
be supplied to training without counting them in every algorithm's budget.
This reference floor is not automatically a lower bound on each learner's
own-occupancy loss or on task-return suboptimality.

If no candidate passes the misspecification gate within the preparation budget,
retain the controlled experiment as certified agnostic evidence and label the
classical results as capacity-restricted empirical tests with unverified
misspecification. Do not silently change expert labels to create a favorable
result.

## Algorithms, optimization, and policy semantics

| Arm | Collection behavior | Training data | Primary fit |
| --- | --- | --- | --- |
| FTL-DAgger | Learner only, beta = 0 | Aggregate retained learner-state labels | Cold unregularized refit |
| BC-iid | Expert only, beta = 1 | Aggregate retained fresh expert-state labels | Same cold unregularized refit |
| Fixed BC | Fixed offline expert dataset | Reuse its selected subset | Same unregularized objective |
| Optional FTRL | Learner only, beta = 0 | Same aggregation rule as FTL | Prespecified regularization, secondary |

Cold refitting means recreating both model weights and optimizer state at each
fit, including momentum/adaptive moments and stopping state. Match loss,
minibatching, convergence criterion, and computational cap across the primary
arms. Record fitting failures and attained objectives. Finite SGD on a nonconvex
model is an approximate implementation of FTL, not an exact offline oracle.
The current BC trainer also defaults to entropy regularization of 0.001.
Expose and set this coefficient to zero for the proposed unregularized primary
comparison, along with zero L2 for FTL and both BC arms. That experiment hook
is proposed work; the baseline-default PR does not silently change its loss.
Do not call FTRL unregularized FTL or allow a secondary FTRL configuration to
replace the primary comparison after results are seen.

CURRENT versus PROPOSED fitting caveats. In the current runner, fixed BC uses
batch size min(32, N) on its N retained samples, while the round-based arms
use min(32, samples_per_round). The first fit uses the policy constructor's
initialization, whereas later cold resets reinitialize trainable parameters
with Xavier uniform weights and zero biases. The current FTL-DAgger versus
BC-iid comparison shares both rules, so it is internally matched. The proposed
comparison that includes fixed BC must resolve both differences before its
optimization can be called matched. This is remaining implementation work.

Classical episodes have variable length. The current per-trajectory sampler
draws an episode and then a uniform time within it, which is an
episode-normalized state distribution, not fixed-horizon occupancy sampling
and not necessarily transition-weighted visitation. Declare this estimand and
use the same one in every arm; do not compare arms whose sampling rules target
different estimands without saying so.

Use deterministic expert labels in the primary experiment. State separately
whether learner behavior samples its action distribution or takes argmax. A
beta of zero removes expert action mixing; it does not by itself make learner
actions deterministic. Use the same prespecified learner behavior convention
for FTL and FTRL. The exact toy uses deterministic class policies.

A stochastic-expert or stochastic-learner experiment is optional and separate.
For stochastic labels, distinguish approximating the conditional label
distribution, its mode, and its mean. Do not interpret unavoidable label noise
as proof that a deterministic expert function is outside the learner class.

Primary practical evaluation is the final trained policy at a fixed data budget.
Also evaluate the theoretical episode-level mixture by drawing one past policy
uniformly at episode start and retaining it for the entire episode. This differs
from resampling a policy/action at each step or averaging neural parameters.
Evaluate and label the two outputs separately; select neither by test return.

## Data budgets and current annotation-cost caveat

The first set of curves should compare equal numbers B of retained labeled
training observations, with candidate checkpoints
{128, 256, 512, 1024, 2048, 4096}. Pilot measurements may justify a different
grid, which must be frozen before confirmation. To limit repeated cold fits,
pilot 16 retained states per round from 16 distinct trajectories. A maximum
budget of 4,096 then needs 256 fits rather than 4,096. Use the same batch size
for FTL-DAgger and BC-iid; keep a small batch-size sensitivity check in the
cheap controlled MDP. Evaluate at the six budget checkpoints with 100 fixed
evaluation episodes per classical checkpoint, rather than at every round.
Record all required policies separately if evaluating the episode-level mixture.
These proposed settings require wiring configurable evaluation budgets into
the runner; they are not its current defaults. Count any common initialization
dataset in B. If no common initialization is used, record the initial model and
its data-free tie-breaking convention instead.

For each checkpoint B, fit fixed BC on B retained examples from a prespecified
offline pool. If fixed BC and BC-iid use the same iid sampling distribution,
their aggregate size-B datasets have the same law. With exact ERM and shared
samples, their fitted outputs should coincide. This is a useful optimization
and scheduling control, not evidence for a distinct statistical principle.

If fixed BC uses temporal prefixes or contiguous trajectories while BC-iid uses
fresh uniformly sampled visits, identify trajectory correlation and state
coverage as differences. Include a matched uniform-selection condition before
attributing an improvement to learner interaction. Replaying a fixed B_0-label
pool while another arm acquires B > B_0 labels is a separate replay diagnostic,
not an equal-label comparison.

The existing trajectory-collection path can call the expert over complete
trajectories before retaining a subset. Its retained-observation x-axis is
therefore not a count of all expert annotations or all environment steps.
Moreover, the name BC-iid describes fresh expert-rollout collection; multiple
retained transitions from the same trajectory need not be independent.

Every result must report at least:

- Retained labeled training observations, including repeated retained states.
- Expert action entries actually requested or produced, including discarded
  labels and actions used to drive expert rollouts. Count entries, not merely
  vectorized prediction-function invocations.
- Environment transitions, completed trajectories, training fits/updates, and
  elapsed time.
- Diagnostic-only expert calls and evaluation costs, separately from training.

Do not double-count one cached expert response when it serves both as an
executed action and a stored label, and do not give one method unreported
caching advantages. Freeze the caching convention across methods.

A proposed deferred-query hook would collect a learner-only trajectory, choose
states for retention, and then ask the expert only for their labels. This can
reduce discarded expert annotations for beta = 0. It is not implemented here.
BC-iid still requires expert actions to generate expert trajectories, including
actions whose transitions are not retained. Deferred storage does not remove
that cost. Report both retained-data curves and total-oracle-action curves;
never relabel the former as the latter. A future equal-total-oracle-budget run
may produce different retained dataset sizes across methods and must say so.

## Statistics, reporting, and negative outcomes

The primary comparison is FTL-DAgger versus BC-iid at the frozen final retained
budget in each selected classical task. Report task return in its native units
with higher-is-better signs stated explicitly. For the toy, report lower-is-
better cost, expert-relative gap, and class-relative excess cost separately.
Use paired training-seed blocks and shared evaluation seed sets when possible.
Evaluation episodes are repeated measurements within a training run, not
independent training replicates.

Target 10 to 20 confirmatory classical training seeds per primary configuration
if measured throughput permits. Choose the final seed count using pilot runtime
and desired interval precision, not whether an effect becomes significant.
Pilot seeds are excluded from confirmation. Use 95% confidence intervals across
training seeds, with paired resampling for differences and episode uncertainty
accounted for within each run when material. Report per-task effects and adjust
the prespecified family of primary tests if making simultaneous significance
claims. Do not use repeated inspection of unadjusted intervals as a stopping rule.

Secondary metrics are on-policy disagreement, common-reference disagreement,
learning-curve area over a frozen log-budget interval, success rate where
defined, and resource consumption. A t-SNE plot can illustrate sampled states
but does not establish coverage, agnosticism, or recoverability. Report failed
or incomplete runs and their prespecified handling rather than dropping them.

Results that weaken or falsify the mechanism include no improvement over
equal-data BC-iid, gains confined to a tie-breaking variant, gains disappearing
with matched optimization, or improved imitation loss without improved task
return. FTL cycling or poor last-iterate performance is a legitimate agnostic
outcome. Preserve these outcomes and avoid selecting only favorable budgets,
seeds, learner restrictions, or environments.

## One-week staged budget

The hard cap is 168 elapsed hours from an authorized execution start. This is an
allocation, not a predicted runtime. No execution is started by this proposal.

| Stage | Planned maximum allocation | Gate before proceeding |
| --- | --- | --- |
| Preparation | First 36 hours | Verify baseline semantics, expert quality, proposed hooks, and accounting |
| Excluded pilot | Next 12 hours | Measure end-to-end collection, fitting, and evaluation; freeze two tasks and grid |
| Confirmation | Next 96 hours | Execute the frozen grid within remaining measured budget |
| Analysis and delivery | Final 24 hours | Validate inventories, uncertainty estimates, plots, and reproducibility records |

Measure representative worst-budget jobs, including cold refits and evaluation.
Estimate the remaining campaign from measured per-job durations and demonstrated
worker throughput, leaving contingency for failures and reruns. Do not infer
parallel speedup from an advertised resource count. If preparation or the pilot
overruns, reduce optional learner classes, FTRL, or extra budget checkpoints
before reducing the primary comparison's replication. Do not extend the cap
implicitly. Fewer than the planned confirmatory seeds should be reported as a
precision limitation, with exploratory status where appropriate.

Freeze a manifest containing source revision, environment and library versions,
expert identifiers, policy representations, normalization, sampling and action
semantics, optimizer reset behavior, budgets, seed lists, evaluation protocol,
and the distinction between retained labels and total oracle actions. Keep
machine connection and private execution-setup information out of tracked
artifacts. Expected deliverables are the exact toy checks, complete run inventory,
per-task paired comparisons, both cost-accounting views, and an explicit list of
unverified assumptions and deferred hooks.
