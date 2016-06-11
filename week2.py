from collections import Counter

def prefix_sum(A):
    sums = []
    cur_sum = 0
    for i in A:
        cur_sum += i
        sums.append(cur_sum)
    return sums

def twosum(A, T):
    i = 0
    j = len(A) - 1
    ans = []
    while i < j:
        cur_sum = A[i] + A[j]
        if cur_sum < T:
            i += 1
        elif cur_sum > T:
            j -= 1
        else:
            ans.append((i,j))
            i += 1
            j -= 1
    return ans

"""
Write a function count_2sum_mod that takes an array A (=a_0, a_1,...)
of integers and an integer K and returns the number of a_i + a_j = 0 (mod K)
where 0 <= i < j < len(A)
"""

def count_2sum_mod(A, k):
    from collections import Counter
    mods = Counter(n % k for n in A)
    total = mods[0] * (mods[0] - 1) // 2
    total += mods[k/2] * (mods[k/2] - 1) // 2 if k % 2 == 0 else 0
    for i in mods:
        if i < k-i:
            total += mods[i] * mods[k-i]
    return total


"""
Write a function count_contig_mod that takes an array A of integers and an
integer k and returns the number of contiguous non-empty subarrays that are 0 mod k.
"""

def count_contig_mod(A, k):
    n = len(A)
    num_sub_arrays = 0
    p_sums = prefix_sum(A)
    for i, s in enumerate(p_sums):
        for j in range(i+1, n):
            if (p_sums[j] - s) % k == 0:
                num_sub_arrays += 1
            j += 1
    return num_sub_arrays


"""
Write a function count_2sum that takes an array A of integers and an integer target
and returns a set of all distinct (a_i, a_j) with a_i <= a_j and a_i + a_j = target.
"""

def count2_sum(A, target):
    counts = Counter(A)
    return set((min(c, target-c), max(c, target-c)) for c in counts if target-c in counts)


"""
Write a function find_intersection which takes two sorted arrays A and B of distinct elements, and
returns a sorted list of elements that are common to both A and B.

a = [1,1,3,4,5,5,6]
b = [2,5,6,7,8,9,2]
result = [5, 6]
"""

def find_intersection(A, B):
    i = j = 0
    intersection = set()
    while i < len(A):
        while B[j] <= A[i]:
            if A[i] == B[j]:
                intersection.add(A[i])
            j += 1
        i += 1
    return intersection


"""
Write a function three_sum which is similar to twosum except you are
searching for i<j<k with A[i] + A[j] + A[k] = target.
[1,2,3,4,5,6], 9 -> (0,1,5), (1,2,3), (0,2,4)
"""

def three_sum(A, target):
    triples = []
    n = len(A)
    for i, num in enumerate(A):
        j = i + 1
        k = n - 1
        while j < k:
            t = target - num
            two_sum = A[j] + A[k]
            if two_sum < t:
                j += 1
            elif two_sum > t:
                k -= 1
            else:
                triples.append((i,j,k))
                j += 1
                k -= 1
    return triples


"""
Write a function is_diff_target which takes a sorted array of integers A and an integer
target, and returns True iff there is some i < j with A[j] - A[i] = target

[0,1,3,3,4,6,7,8,9,10], 8 -- 10-0 > 8
"""

def is_diff_target(A, target):
    n = len(A)
    i = n - 2
    j = n - 1
    while (A[j] - A[i]) < target:
        i -= 1
    while i >= 0 and j >= 0:
        if (A[j] - A[i]) > target:
            j -= 1
        elif (A[j] - A[i] < target):
            i -= 1
        else:
            return True
    return False


"""
Write a function prefix_sum_2d which takes a 2d array A and a list of queries Q
where each query is of the form (r1,c1,r2,c2) representing indices for the topleft
and bottomright row and column of some subrectangle of A.  Return a list where
each element is the the sum of the corresponding subrectangle in A for each query.
(Your program should answer these queries in constant time after some preprocessing.)
"""

def prefix_sum_2d(A, queries):
    if A == []: return []
    z = zip(*A)
    col_prefix_sums = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    for i, column in enumerate(z):
        col_sum = 0
        for j, cell in enumerate(column):
            col_sum += cell
            col_prefix_sums[j][i] = col_sum

    rect_sums = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    for i, row in enumerate(A):
        for j, col in enumerate(row):
            prev = rect_sums[i][j-1] if j > 0 else 0
            rect_sums[i][j] = prev + col_prefix_sums[i][j]

    query_results = []
    for q in queries:
        r1, c1, r2, c2 = q
        common_sum = rect_sums[r1-1][c1-1] if r1 > 0 and c1 > 0 else 0
        prev_col_sum = rect_sums[r2][c1-1] if c1 > 0 else 0
        prev_row_sum = rect_sums[r1-1][c2] if r1 > 0 else 0
        result = rect_sums[r2][c2] - prev_col_sum - prev_row_sum + common_sum
        query_results.append(result)
    return query_results


"""
Write a function powerset which returns the powerset of a list A.
(The powerset is all 2^len(A) subsets of A.)
"""

def powerset(A):
    from copy import deepcopy
    if A == []: return [[]]
    else:
        elt = A[-1]
        prev = powerset(A[:-1])
        new = deepcopy(prev)
        for i in range(len(new)):
            new[i].append(elt)
        prev.extend(new)
        return prev

"""
Write a function that finds the second maximum element of an array of integers in linear time.
"""

def second_max(A):
    maximum = max(A)
    second_maximum = None
    for elt in A:
        if (elt != maximum) and (second_maximum is None or elt > second_maximum):
            second_maximum = elt
    return second_maximum

# Print statements to check output

print (count_2sum_mod([3,4,4,5,6,6,6], 9))
print (count_2sum_mod([0,0,0,1,1,1,2,2,2,3,4], 4))
print (count_contig_mod([4,5,5,1,4,5,8], 5))
print (count2_sum([3,4,3,2,4,0,1,2,4,2,6,3,3], 6))
print (find_intersection([1,1,3,4,5,5,6], [2,2,5,6,7,8,9]))
print (three_sum([1,2,3,4,5,6], 9))
print (is_diff_target([1,3,3,3,4,6,7,8.5,10,12], 1))
print (is_diff_target([0,4,7,8,9,10,19,20,24], 1))
print (prefix_sum_2d([[5,6,3,7,8],
                     [1,9,2,8,6],
                     [11,10,5,4,2],
                     [8,5,2,3,6]],
                    [(1,2,3,3), (0,0,1,1)]))
print (powerset([1,2,3]))
print (len(powerset([1,2,3,4,5])))
print (second_max([7,3,3,7,6,4,2]))

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

"""
[1,2,3,4], 2 -> [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]
[1], [2], [3], [4]
"""
def combinations(A, K):
    if not K: return [[]]
    if not A: return []
    head = [A[0]]
    tail = A[1:]
    chosen = [head + lt for lt in combinations(tail, K-1)]
    chosen.extend(combinations(tail, K))
    return chosen

def insert(elt, lst):
    result = []
    for i in range(len(lst)+1):
        new = lst[:]
        new.insert(i, elt)
        result.append(new)
    return result

def permutations(A):
    if not A: return [[]]
    head = A[0]
    tail = A[1:]
    result = []
    p = permutations(tail)
    for each in p:
        result.extend(insert(head, each))
    return result

def permutations2(A):
    if len(A) == 0: return []
    elif len(A) == 1: return [A]
    else:
        l = []
        for i in range(len(A)):
            elt = A[i]
            rest = A[:i] + A[i+1:]
            for p in permutations2(rest):
                l.append([elt]+p)
        return l

def increment_num(lst):
    number = 0
    for i, n in enumerate(reversed(lst)):
        number += n * (10**i)
    return number + 1

print increment_num([9,9])

def water_trapped(heights):
    total = 0
    larger_before = preprocess(heights)
    larger_after = preprocess(list(reversed(heights)))
    for i in range(len(heights)):
        amount = min(larger_before[i], larger_after[-i-1]) - heights[i]
        total += amount if amount > 0 else 0
    return total

def preprocess(heights):
    larger_before = []
    highest = 0
    for j in range(len(heights)):
        larger_before.append(highest)
        if heights[j] > highest:
            highest = heights[j]
    return larger_before

print "water amount = " + str(water_trapped([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))