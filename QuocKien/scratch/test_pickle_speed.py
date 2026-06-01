import time
import pickle

def test():
    customer_ids = list(range(100000))
    items_list = [["item_A", "item_B", "item_C", "item_D", "item_E", "item_F", "item_G", "item_H", "item_I", "item_J"] for _ in range(100000)]
    
    start = time.time()
    for cid, items in zip(customer_ids, items_list):
        k = pickle.dumps(cid, protocol=4)
        v = pickle.dumps(items, protocol=4)
    end = time.time()
    print(f"Time for 100k pickle dumps: {end - start:.2f} seconds")

if __name__ == '__main__':
    test()
