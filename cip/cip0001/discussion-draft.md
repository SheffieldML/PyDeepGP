**Category:** Ideas
**Title:** CIP-0001: Replacing Theano and MPI in PyDeepGP with modern NumPy. Do you use MPI?

---

PyDeepGP's default model no longer builds on a current Python stack. `DeepGP(...)` uses the back-constrained MLP encoder by default, and that encoder needs **Theano**, which is unmaintained and won't install on Python 3.13. The **MPI** path also crashes on NumPy ≥ 1.24. With `back_constraint=False` it runs, but gives silently wrong results.

**[CIP-0001](../cip0001.md)** proposes:

1. **A NumPy encoder** to replace Theano. A prototype reproduces the stored reference log-likelihoods for the back-constrained models (those tests are skipped in CI today) and passes all gradient checks.
2. **Fixing MPI, then deprecating it.** Fix the crash, and turn the silently wrong configuration into a clear error.
3. **Chunked psi-statistics in NumPy**, optionally threaded, to replace MPI on a single machine. On a 2-core test machine, 2 threads matched 2 MPI ranks in speed and used roughly 4–7× less memory.
4. **Removing MPI** after a deprecation window, unless people need multi-machine runs.

**Questions for users:**

- **Do you run PyDeepGP with MPI across several machines?** This decides whether MPI is removed or kept in a minimal form.
- **Do you pass `regularization` / `reg_weight` to the encoder?** Those options currently have no effect. Should the fix keep them as they are (with a warning) or make them work as documented, which would change results?
- **Do you have access to a many-core machine** and could run the chunked psi-statistics benchmark? We've only measured scaling on 2 cores.

Comments on the CIP itself are welcome here too.
