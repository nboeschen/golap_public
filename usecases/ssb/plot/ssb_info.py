import numpy as np


LO_TUPLE = 120
CUST_TUPLE = 152
PART_TUPLE = 144
SUPP_TUPLE = 136
DATE_TUPLE = 120

def info(scale_factor, customer_factor=100):
    total = [0]
    def p(total,num,size):
        total[0] += (num*size) / (1<<30)
        return f"{num} tuples, {(num*size) / (1<<30):.3f} GB, {(num*size)}"

    print("Lineorder:",p(total,scale_factor * 1500000 * 5, LO_TUPLE))
    print("Customer :",p(total,scale_factor * 30000 * customer_factor, CUST_TUPLE))
    print("Part     :",p(total,int(200000*np.floor(1+np.log2(scale_factor))), PART_TUPLE))
    print("Supplier :",p(total,scale_factor * 2000, SUPP_TUPLE))
    print("Date     :",p(total,365*7, DATE_TUPLE))
    print(f"Total    : {total[0]:.2f}GB")


if __name__ == '__main__':
    info(100,1)
