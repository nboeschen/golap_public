import numpy as np


ORDER_TUPLE = 85
LINEITEM_TUPLE = 128
PART_TUPLE = 180
SUPP_TUPLE = 136
PARTSUPP_TUPLE = 28
CUST_TUPLE = 148
NATION_TUPLE = 32
REGION_TUPLE = 24

def info(scale_factor, with_comment):
    total = [0]
    def p(total,num,size):
        total[0] += (num*size) / (1<<30)
        return f"{num} tuples, {(num*size)} B, {(num*size) / (1<<30):.3f} GB"

    print("ORDER: ",        p(total,scale_factor*1500000, ORDER_TUPLE))
    print("LINEITEM: ",     p(total,scale_factor*1500000*5, LINEITEM_TUPLE))
    print("PART: ",         p(total,scale_factor*200000, PART_TUPLE))
    print("SUPP: ",         p(total,scale_factor*10000, SUPP_TUPLE))
    print("PARTSUPP: ",     p(total,scale_factor*800000, PARTSUPP_TUPLE))
    print("CUST: ",         p(total,scale_factor*150000, CUST_TUPLE))
    print("NATION: ",       p(total,25, NATION_TUPLE))
    print("REGION: ",       p(total,5, REGION_TUPLE))
    print(f"Total    : {total[0]:.2f}GB")



if __name__ == '__main__':
    info(200,with_comment=False)
