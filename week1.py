def double_char(inp):
    return ''.join(s * 2 for s in inp)


def contains_true_love(inp):
    return inp.find('.love') == -1 and inp.find('love') != -1


def palindrome_index(inp):
    def is_palindrome(string):
        head = 0
        tail = len(string) - 1
        while head < tail:
            if string[head] != string[tail]:
                return False
            head += 1
            tail -= 1
        else:
            return True

    start = 0
    end = len(inp) - 1
    while start < end:
        if inp[start] != inp[end]:
            if is_palindrome(inp[:start] + inp[start + 1:]):
                return start
            else:
                return end
        start += 1
        end -= 1
    else:
        return -1


def my_zip(*iters):
    n = len(min(iters, key = lambda x: len(x)))
    zipped = [[iters[j][i] for j in range(len(iters))] for i in range(n)]
    return [tuple(z) for z in zipped]


def least_flips(bin_list, k):
    n = len(bin_list)
    least_zeros = num_zeros = k - sum(bin_list[0:k])
    for i in range(1, n - k + 1):
        if bin_list[i-1] == 0:
            num_zeros -= 1
        if bin_list[i+k-1] == 0:
            num_zeros += 1
        if num_zeros < least_zeros:
            least_zeros = num_zeros
    return least_zeros


def test_least_flips():
    assert least_flips([], 0) == 0
    assert least_flips([0], 1) == 1
    assert least_flips([0], 1) == 1
    assert least_flips([1,0,0,0,1,1,0,0,1], 4) == 2
    assert least_flips([0, 0, 0, 1, 0, 1, 1, 0, 0, 1], 2) == 0
    assert least_flips([1,0,1,0,1,0,1,0,0,0,0,1,1,1,1], 7) == 3
    assert least_flips([0, 0, 0, 1, 0, 1, 1, 0, 0, 1], 3) == 1
    print ("Tests pass!!")


def stock_rmas(L):
    current_sum = sum(L[i] for i in range(30))
    current_avg = sum((i + 1) * L[i] for i in range(30)) / 30.0
    n = len(L)
    averages = [current_avg]
    for i in range(0, n - 30):
        current_avg -= current_sum / 30.0
        current_avg += L[i + 30]
        current_sum += L[i + 30] - L[i]
        averages.append(current_avg)
    return averages


def my_pow(base, exp, modulus):
    from datastructures import Stack
    powers = Stack()
    var_exp = exp
    while var_exp > 0:
        powers.push(var_exp % 2)
        var_exp = var_exp / 2
    rem = 1
    while not powers.is_empty():
        p = powers.pop()
        rem = ((base ** p) * ((rem ** 2) % modulus)) % modulus
    return rem