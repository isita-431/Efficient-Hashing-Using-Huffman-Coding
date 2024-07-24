import sqlite3
import zlib
import heapq
import random
import string
from memory_profiler import profile
from memory_profiler import memory_usage
import nltk
from nltk.corpus import gutenberg
from nltk import word_tokenize
import matplotlib.pyplot as plt

class HuffmanNode:
    def __init__(self, key, frequency):
        self.key = key
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

class HuffmanTree:
    def __init__(self):
        self.root = None

    def build_tree(self, frequencies):
        heap = [HuffmanNode(key, freq) for key, freq in frequencies.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            merged_node = HuffmanNode(None, left.frequency + right.frequency)
            merged_node.left = left
            merged_node.right = right

            heapq.heappush(heap, merged_node)

        self.root = heap[0]

    def generate_codes(self, node, current_code, codes):
        if node.key is not None:
            codes[node.key] = current_code
        if node.left:
            self.generate_codes(node.left, current_code + "0", codes)
        if node.right:
            self.generate_codes(node.right, current_code + "1", codes)

def compress_data(data):
    return zlib.compress(data.encode('utf-8'))

def decompress_data(compressed_data):
    return zlib.decompress(compressed_data).decode('utf-8')

class ExtendibleHashing:
    def __init__(self, global_depth=2):
        self.global_depth = global_depth
        self.directory = {}

    def hash_function(self, key):
        hash_value = hash(key)
        return hash_value % (2**self.global_depth)

    def insert(self, key, value):
        hash_value = self.hash_function(key)
        if hash_value not in self.directory:
            self.directory[hash_value] = []

        self.directory[hash_value].append((key, value))

    def print_directory(self):
        for index, bucket in self.directory.items():
            print(f"Bucket {index}: {bucket}")

def generate_random_string(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def generate_larger_dataset(size):
    # Use words from Shakespeare's "Hamlet" as a source
    words = gutenberg.words('shakespeare-hamlet.txt')
    random_words = [random.choice(words) for _ in range(size)]
    return ' '.join(random_words)

# Function to test Extendible Hashing without Huffman coding and larger data
@profile
def test_extendible_hashing_without_huffman_and_larger_data(data_size):
    extendible_hashing = ExtendibleHashing(global_depth=2)

    larger_data = generate_larger_dataset(data_size)

    for word in word_tokenize(larger_data):
        extendible_hashing.insert(word, word)

# Function to test Extendible Hashing with Huffman coding and variable data size
@profile
def test_extendible_hashing_with_huffman_and_larger_data(data_size):
    extendible_hashing = ExtendibleHashing(global_depth=2)

    larger_data = generate_larger_dataset(data_size)

    huffman_tree = HuffmanTree()
    huffman_tree.build_tree({word: len(word) for word in word_tokenize(larger_data)})
    codes = {}
    huffman_tree.generate_codes(huffman_tree.root, "", codes)

    for word, code in codes.items():
        compressed_value = compress_data(code)
        extendible_hashing.insert(word, compressed_value)



if __name__ == '__main__':


    def plot():

        data_sizes = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
        memory_usage_without_huffman = []
        memory_usage_with_huffman = []
        
        for data_size in data_sizes:
            memory_used_without_huffman = memory_usage((test_extendible_hashing_without_huffman_and_larger_data, (data_size,), {}))
            memory_used_with_huffman = memory_usage((test_extendible_hashing_with_huffman_and_larger_data, (data_size,), {}))

            memory_usage_without_huffman.append((memory_used_without_huffman[-1]-memory_used_without_huffman[0]))
            memory_usage_with_huffman.append((memory_used_with_huffman[-1]-memory_used_with_huffman[0]))
            
        print(memory_usage_with_huffman)
        print(memory_usage_without_huffman)
        plt.plot(data_sizes, memory_usage_without_huffman, label='Without Huffman Coding')
        plt.plot(data_sizes, memory_usage_with_huffman, label='With Huffman Coding')
        plt.xlabel('Data Set Size')
        plt.ylabel('Memory Usage (MiB)')
        plt.legend()
        plt.title('Memory Usage Comparison: Extendible Hashing with and without Huffman Coding')
        plt.show()

    plot()
