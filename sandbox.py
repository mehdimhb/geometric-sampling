from fast.algorithm import Design, generate_design

# TODO
# make bars out of design
# make sample from design
x = [0.07, 0.04, 0.03, 0.06] * 5_000 + [0.02] * 5_000
print(sum(x), len(x))
d = Design(x)
# d.show()
dd = generate_design(d, 100)
print(len(dd.heap))
dd.merge_identical()
print(len(dd.heap))
# dd.show()