



t_all.sort()
for dt in t_all:
    o = orders[(orders.date==dt) and (orders.priority = 1)]
    o.drop(labels = ['date','marination'], axis = 1, inplace= True)
    i = parts_inv
