# Permuted target notebook verification

Verification only, not the full configured experiment.

The executed Jupyter copy uses N=12, n=4, unequal FIP, SEED=12345, 20 accepted exchanges from a separately built permuted-order standard GFS design. The permutation is [7, 10, 1, 4, 3, 8, 11, 6, 0, 2, 9, 5]. Both optimizers start from the standard original-order design. Directed uses three independent seeds, 2,000 candidate directions each and two distinct child worker PIDs.

Assertions check the permutation seed, both initial orders, restoration to original-ID SIP order, independently recomputed target SIP, fixed sample size, prescribed FIP, original-start optimizer trace errors and consistency of saved seeds/permutation/results. The target's max FIP error was 2.22e-16; original-start-to-target RMSE=0.1380209903 and NRMSE=1.192053228. These values belong only to this verification problem.

Saved verification data: `prescribed_sip/results/notebook_directed/N12_n4_20260918T161511_738223Z/`.
