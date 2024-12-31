# Project Euler Code
from extraFunctions import primeSieve, EuclideanAlg, timeTest, isPalindrome, divisor_count, is_prime
from math import log, ceil, comb, factorial, sqrt, floor, gcd
from gmpy2 import mpz
import networkx as nx
import itertools
import os
import time


def problem_001(n=1000):
    """
    Find the sum of all the multiples of 3 or 5 below n.
    """
    def triangle(n):
        # Sum of 0 to n is given by
        return (n * n + n)//2
    # How many multiples of 3 are there? We can then factor the sum to
    # 3 + 6 + 9 ... = 3 (1 + 2 + ... ) using Triangle
    n_3 = n//3  
    n_5 = n//5  
    n_15 = n//15  
    # Sum the 3s and 5s, subtract what we double count
    fizz_buzz_sum = triangle(n_3) + triangle(n_5) - triangle(n_15)
    return fizz_buzz_sum


def problem_002(n=4000000):
    """
    By considering the terms in the Fibonacci sequence whose values do not exceed n,
    find the sum of the even-valued terms.
    """
    fib = [0, 1]
    testVal = 1
    while testVal <= n:
        fib.append(testVal)
        testVal = fib[-1] + fib[-2]
    # F_n is even when n%3==0 
    # Proof: Consider F_n % 2: [0, 1, 1, 0, 1, 1 ...]
    evenSum = 0
    for i in range(0, len(fib), 3):
        evenSum += fib[i]
    return evenSum


def problem_003(n=600851475143):
    """
    What is the largest prime factor of the n?
    FAST FOR HIGHLY COMPOSITE N
    """
    # Iteratively divide by 2
    while n % 2 == 0:
        n = n//2
    if n == 1:
        return 2
    """
    # Naive implementation likely fastest -- tested for values in (1,000,000 to 1,005,000)
    # Alternative methods: Bit Shifting
    while n & 1 == 0:  # Check if the least significant bit is 0
        n >>= 1
    
    # Alternative methods: Stripping trailing zeros
    b = str(bin(n))
    n = int(b.rstrip("0"), 2)
    """
    # Iterate through odd numbers, largest divisor is largest prime factor.
    p = 1
    while n != 1:
        p += 2
        while n % p == 0:
            n = n//p
    return p


def problem_004(n=3):
    """
    Find the largest palindrome made from the product of two n-digit numbers
    """
    minVal = 10**(n-1)
    maxVal = 10**n
    maxPalindrome = 1
    for i in range(minVal, maxVal):
        for j in range(i, maxVal):
            x = i*j
            if x > maxPalindrome and isPalindrome(x):
                maxPalindrome = x
    return maxPalindrome


def problem_005(n=20):
    """
    What is the smallest integer that is divisible by all integers from 1 to n?
    """
    primes = primeSieve(n)
    composite = 1
    for prime in primes:
        k = prime
        while k*prime < n:
            k *= prime
        composite *= k
    return composite


def problem_006(n=100):
    """
    Find the difference between the sum of the squares of the first n numbers and the square of the sum.
    """
    # Faulhaber formulas used for power = 1, 2
    sum_of_n = n*(n+1)//2
    sum_of_squares = n*(n+1)*(2*n+1)//6
    return sum_of_n**2 - sum_of_squares


def problem_007(n=10001):
    """
    What is the n-th prime number?
    """
    upper_bound = ceil(n*2*log(n))
    primes = primeSieve(upper_bound)
    return primes[n-1]


def problem_008(n=13):
    """
Find the n adjacent digits in the 1000-digit number that have the greatest product. What is the value of this product?
    """
    filename = "data/0008_number.txt"
    s = ""
    with open(filename) as file:
        for line in file:
            s += line.strip(",\n")
    max_val = 0
    for i in range(len(s)-n):
        prod = 1
        for j in range(n):
            prod *= int(s[i+j])
            if prod > max_val:
                max_val = prod
    return max_val


def problem_009(n=1000):
    """
    There exists exactly one Pythagorean triplet for which a+b+c=1000. Find the product abc.
    Generalized to a+b+c=n
    """
    solution = []
    for c in range(n//3, n):
        for b in range(1, c+1):
            a = n - b - c
            if a < b:
                continue
            if a**2 + b**2 == c**2:
                solution.append(a*b*c)
    return solution


def problem_010(n=2000000):
    """
    Find the sum of all the primes below n.
    """
    primes = primeSieve(n)
    sum_primes = sum(primes)
    return sum_primes


def problem_011(n=4):
    """
    What is the greatest product of n adjacent numbers in the same direction
    (up, down, left, right, or diagonally) in the 20x20 grid?
    """
    # Read file
    filename = "data/0011_grid.txt"
    grid = []
    with open(filename) as file:
        for line in file:
            row = line.split(" ")
            grid.append([int(val) for val in row])
    # Find max product along rows and columns
    max_product = 0
    for i in range(len(grid)-n):
        for j in range(len(grid[i])):
            # rows
            prod = 1
            for p in range(n):
                prod *= grid[i+p][j]
            if prod > max_product:
                max_product = prod
            # columns
            prod = 1
            for p in range(n):
                prod *= grid[j][i+p]
            if prod > max_product:
                max_product = prod
    # Find max product along diagonals
    for i in range(len(grid)-n):
        for j in range(len(grid[i]) - n):
            # positive diagonal
            prod = 1
            for p in range(n):
                prod *= grid[i+p][j+p]
            if prod > max_product:
                max_product = prod
            # negative diagonal
            prod = 1
            for p in range(n):
                prod *= grid[i+n-1-p][j+p]
            if prod > max_product:
                max_product = prod
    return max_product


def problem_012(n=500):
    """
    What is the value of the first triangle number to have over n divisors?
    """
    primes = primeSieve(5000000)
    triangle = 1
    next_row = 2
    while divisor_count(triangle, primes) < n:
        if next_row > 10**8:
            break
        triangle += next_row
        next_row += 1
    return triangle


def problem_013(n=10):
    """
    Work out the first n digits of the sum of the following one-hundred 50-digit numbers.
    """
    filename = "data/0013_numbers.txt"
    nums = []
    with open(filename) as file:
        nums = [int(line.strip(",\n")) for line in file]
    sum_nums = sum(nums)
    digits = str(sum_nums)[:n]
    return digits


def problem_014(n=1000000):
    """
    Which starting number, under n, produces the longest Collatz chain?
    """
    collatz_dict = {1: 1}
    max_chain = 1
    for num in range(2, n+1):
        k = [num]
        while k[-1] not in collatz_dict:
            if k[-1] % 2 == 0:
                k.append(k[-1]//2)
            else:
                k.append(3*k[-1] + 1)
        for i in range(len(k)):
            if k[i] <= n:
                val = collatz_dict[k[-1]] + len(k) - i - 1
                collatz_dict[k[i]] = val
                if val > collatz_dict[max_chain]:
                    max_chain = k[0]
    return max_chain


def problem_015(m=20, n=20):
    """
    A valid route starts in the top left and traverses to the bottom 
    right by only moving only right and down.
    How many such routes are there through a mxn grid?
    """
    val = comb(m+n, n)
    return val


def problem_016(n=1000):
    """
    What is the sum of the digits of the number 2**n?
    """
    num = 2**n
    s = str(num)
    total = 0
    for i in s:
        total += int(i)
    return total


def problem_017(n=1000):
    """
    If all the numbers from 1 to n inclusive were written out in words,
    how many letters would be used?
    """
    # The length for small numbers in English
    word_dict = {0: 0, 1: 3, 2: 3, 3: 5, 4: 4, 5: 4, 6: 3, 7: 5, 8: 5, 9: 4,
                 10: 3, 11: 6, 12: 6, 13: 8, 14: 8, 15: 7, 16: 7, 17: 9, 18: 8, 19: 8,
                 20: 6, 30: 6, 40: 5, 50: 5, 60: 5, 70: 7, 80: 6, 90: 6}
    letter_sum = 0
    for i in range(1, n+1):
        word_sum = 0
        num = i
        if num >= 1000:
            t = num // 1000
            word_sum += word_dict[t] + len("thousand")
            num = i % 1000
        if num >= 100:
            # Hundred protocol
            h = num // 100
            word_sum += word_dict[h] + len("hundred")
            num = i % 100
            if num > 0:
                word_sum += len("and")
        if num >= 20:
            t = num // 10
            t *= 10
            word_sum += word_dict[t]
            num -= t
        word_sum += word_dict[num]
        letter_sum += word_sum
    return letter_sum


def problem_018():
    """
    Find the maximum total from top to bottom of the triangle below
    """
    # Read file
    filename = "data/0018_pyramid.txt"
    pyramid= []
    with open(filename) as file:
        for line in file:
            row = line.split(" ")
            pyramid.append([int(val) for val in row])
    # Determine best path by starting at the bottom
    for row in range(len(pyramid)-2, -1, -1):
        for item in range(len(pyramid[row])):
            pyramid[row][item] = int(pyramid[row][item])
            pyramid[row][item] += max(pyramid[row+1][item], pyramid[row+1][item+1])
    max_val = pyramid[0][0]
    return max_val


def problem_019():
    """
    How many Sundays fell on the first of the month during the twentieth century
    """
    year_day = 0  # Monday is zero
    year_dict = {0: 2, 1: 2, 2: 1, 3: 3, 4: 1, 5: 1, 6: 2}
    leap_dict = {0: 1, 1: 2, 2: 2, 3: 1, 4: 1, 5: 2, 6: 3}
    sundays = 0
    for year in range(0, 101):
        if year % 4 == 0 and year != 0:
            sundays += leap_dict[year_day]
            year_day += 2
            year_day = year_day % 7
        else:
            if year != 0:
                sundays += year_dict[year_day]
            year_day += 1
            year_day = year_day % 7
    return sundays


def problem_020(n=100):
    """
 Find the sum of the digits in the number n!
    """
    s = str(factorial(n))
    total = 0
    for i in s:
        total += int(i)
    return total


def problem_021(n=10000):
    """
    Evaluate the sum of all the amicable numbers under n
    """
    amicable_sum = 0
    divisors = []
    for i in range(n):
        divisor_sum = 1
        for divisor in range(2, floor(sqrt(i))+1):
            if i % divisor == 0:
                divisor_sum += divisor
                if i // divisor != divisor:
                    divisor_sum += i // divisor
        divisors.append(divisor_sum)
        if divisor_sum < i:
            if divisors[divisor_sum] == i:
                amicable_sum += divisor_sum
                amicable_sum += i
    return amicable_sum


def problem_022():
    """
    What is the total of all the name scores in the file?
    Score is calculated by (index of alphabetized name) x (sum of letters)
    """
    filename = "data/0022_names.txt"
    total = 0
    letter_score = {"AABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]: i for i in range(27)}
    with open(filename) as file:
        temp = [line.strip("\"").split("\",\"") for line in file]
        names = sorted(temp[0])
    for i in range(len(names)):
        word_score = 0
        for letter in names[i]:
            if letter in letter_score:
                word_score += letter_score[letter]
        total += word_score * (i + 1)
    return total


def problem_023():
    """
    Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers.
    """
    upper_bound = 28124 # This is the largest value
    abundants = []
    for i in range(1, upper_bound):
        divisor_sum = 1
        for divisor in range(2, floor(sqrt(i))+1):
            if i % divisor == 0:
                divisor_sum += divisor
                if i // divisor != divisor:
                    divisor_sum += i // divisor
        if divisor_sum > i:
            abundants.append(i)
    two_abundants = set(abundants[i] + abundants[j] for i in range(len(abundants)) for j in range(i, len(abundants)))
    extras = set(i for i in range(1, upper_bound)) - two_abundants
    abundant_sum = sum(extras)
    return abundant_sum


def problem_024(n=1000000, d=10):
    """
    What is the nth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?
    """
    digits = [i for i in range(d)]
    val = []
    num = n - 1
    for i in range(d-1):
        val.append(num // factorial(d-1-i))
        num = num % factorial(d-1-i)
    val.append(0)
    solution = []
    for i in range(d):
        solution.append(str(digits.pop(val[i])))
    solution = int("".join(solution))
    return solution


def problem_025(n=1000):
    """
    What is the index of the first term in the Fibonacci sequence to contain n digits?
    """
    fib = [1, 1]
    while len(str(fib[-1])) < n:
        fib.append(fib[-1] + fib[-2])
    return len(fib)


def problem_026(n=1000):
    """
    Find the value of d < n for which 1/d contains the longest recurring cycle in its decimal fraction part
    """
    max_repeat = 0
    max_val = 0
    for d in range(1, n+1):
        remainders = [1]
        dividing = True
        i = 1
        while dividing:
            val = remainders[-1] * 10 % d
            if val in remainders:
                dividing = False
                repeat = i - remainders.index(val)
                if repeat > max_repeat:
                    max_repeat = repeat
                    max_val = d
            elif val == 0:
                break
            else:
                remainders.append(val)
                i += 1
    return max_val


def problem_027(n=1000):
    """
    Find the product of the coefficients, a and b, for the quadratic expression that produces
    the maximum number of primes for consecutive values of x, starting with x=0. |a|, |b| < n
    """
    primes = set(primeSieve(n**2))
    max_primes = 0
    ab = 0
    for a in range(-n, n+1):
        for b in range(-n, n+1):
            i = 0
            while i**2 + a*i + b in primes:
                i += 1
            if i > max_primes:
                max_primes = i
                ab = a * b
    return ab


def problem_028(n=1001):
    """
    What is the sum of the numbers on the diagonals in a n x n Ulam spiral centered at 1.
    """
    diagonal_sum = 1
    for i in range(1, (n+1)//2):
        diagonal_sum += 4 * (4*i**2 + i + 1)
    return diagonal_sum


def problem_029(n=100):
    """
    How many distinct terms are in the sequence generated by a^b for a,b \in [2, 100]
    """
    power_set = set(a**b for a in range(2, n+1) for b in range(2, n+1))
    return len(power_set)


def problem_030(n=5):
    """
    Find the sum of all the numbers that can be written as the sum of n-th powers of their digits.
    """
    power_dict = {str(i): i**n for i in range(10)}
    upper_limit = 10*power_dict['9']
    total = 0
    for i in range(10, upper_limit):
        num = str(i)
        num_sum = sum([power_dict[j] for j in num])
        if i == num_sum:
            total += i
    return total


def problem_031(n=200):
    """
    How many different ways can $n be made using any number of coins?
    """
    coins = [1, 2, 5, 10, 20, 50, 100, 200]
    dynamic_array = [[0 for i in range(n+1)] for coin in coins]
    for coin in range(len(coins)):
        for cent in range(n+1):
            if coin == 0:
                dynamic_array[coin][cent] = 1
            else:
                dynamic_array[coin][cent] = dynamic_array[coin-1][cent]
                if cent >= coins[coin]:
                    dynamic_array[coin][cent] += dynamic_array[coin][cent - coins[coin]]
    return dynamic_array[-1][-1]


def problem_032():
    """
    Find the sum of all products whose multiplicand/multiplier/product
    identity can be written as a 1 through 9 pandigital.
    """
    total = 0
    found = []
    zero = set('0')
    for i in range(100):
        for j in range(i+1, 2000):
            ij = i * j
            num_str = str(i) + str(j) + str(ij)
            digit_list = list(d for d in num_str)
            digit_set = set(d for d in num_str)
            digit_set = digit_set - zero
            if len(digit_set) == 9 and len(digit_list) == 9:
                if ij not in found:
                    total += ij
                    found.append(ij)
    return total


def problem_033(n=2):
    """
    Find non-trivial examples of digit-cancelling fractions, less than one in value,
    containing n digits in the numerator and denominator.

    If the product of these fractions is given in its lowest common terms, find the value of the denominator.
    """
    total_numer = 1
    total_denom = 1
    for denom in range(11, 10**n):
        for numer in range(10, denom):
            numer_set = set(s for s in str(numer))
            denom_set = set(s for s in str(denom))
            combo = numer_set & denom_set
            reduced_numer = list(str(numer))
            reduced_denom = list(str(denom))
            for i in combo:
                reduced_numer.pop(reduced_numer.index(i))
                reduced_denom.pop(reduced_denom.index(i))
            if len(reduced_numer) > 0 and len(combo) > 0:
                reduced_numer = int("".join(reduced_numer))
                reduced_denom = int("".join(reduced_denom))
                original_gcd = gcd(numer, denom)
                reduced_gcd = gcd(reduced_numer, reduced_denom)
                if numer // original_gcd == reduced_numer // reduced_gcd:
                    if denom // original_gcd == reduced_denom // reduced_gcd:
                        if original_gcd % 10 != 0:
                            total_numer *= numer // original_gcd
                            total_denom *= denom // original_gcd
    total_gcd = gcd(total_numer, total_denom)
    total_numer = total_numer // total_gcd
    total_denom = total_denom // total_gcd
    return total_numer, total_denom


def problem_034():
    """
    Find the sum of all numbers which are equal to the sum of the factorial of their digits.
    """
    factorial_dict = {str(i): factorial(i) for i in range(10)}
    total = 0
    for i in range(10, factorial(9)*10):
        num = i
        digit_total = 0
        for d in str(num):
            digit_total += factorial_dict[d]
        if digit_total == i:
            total += i
    return total


def problem_035(n=1000000):
    """
    How many circular primes are there below n?
    """
    primes = primeSieve(n)
    prime_set = set(primes)
    cyclic_count = 0
    for prime in primes:
        prime_list = list(str(prime))
        cyclic = True
        for d in range(len(prime_list)):
            prime_list.append(prime_list.pop(0))
            new_p = int(''.join(prime_list))
            if new_p not in prime_set:
                cyclic = False
        if cyclic:
            cyclic_count += 1
    return cyclic_count


def problem_036(n=6):
    """
    Find the sum of all numbers, less than 10**n, which are palindromic in base 10 and base 2.
    """
    total_sum = 0
    for i in range(1, 10**((n+1)//2)):
        # Guaranteed palindromes in base 10 have two cases: Even or odd length
        even_len_list = list(str(i))
        num_len = len(str(i))
        for d in range(num_len-1, -1, -1):
            even_len_list.append(even_len_list[d])
        odd_len_list = [d for d in even_len_list]
        odd_len_list.pop(num_len)
        # Test even length palindromes
        even_len_num = int(''.join(even_len_list))
        binary_list = list(bin(even_len_num))
        binary_list.pop(0)
        bin_palindrome = True
        for b in range(1, len(binary_list)):
            if binary_list[b] != binary_list[-b]:
                bin_palindrome = False
        if bin_palindrome:
            total_sum += even_len_num
        # Test odd length palindromes
        odd_len_num = int(''.join(odd_len_list))
        binary_list = list(bin(odd_len_num))
        binary_list.pop(0)
        bin_palindrome = True
        for b in range(1, len(binary_list)):
            if binary_list[b] != binary_list[-b]:
                bin_palindrome = False
        if bin_palindrome:
            total_sum += odd_len_num
    return total_sum


def problem_037():
    """
    Truncatable primes remain prime when removing digits from left to right. eg 3797
    Find the sum of the only eleven primes that are both truncatable
    from left to right and right to left.
    NOTE: 2, 3, 5, 7 are not considered truncatable.
    """
    n = 1000000
    primes = primeSieve(n)
    prime_set = set(primes)
    truncated_sum = 0
    for i in range(4, len(primes)):
        p = str(primes[i])
        truncated_prime = True
        for j in range(1, len(p)):
            leftPrime = int(p[:-j])
            rightPrime = int(p[j:])
            if leftPrime in prime_set and rightPrime in prime_set:
                continue
            else:
                truncated_prime = False
        if truncated_prime:
            truncated_sum += int(p)
    return truncated_sum


def problem_038():
    """
    What is the largest 1-9 pandigital 9-digit number that can be formed as
     the concatenated product of an integer with (1, 2, ..., n) where n > 1?
    """
    max_concat = 0
    for i in range(10, 100000):
        str_i = str(i)
        n = 2
        while len(str_i) < 9:
            str_i += str(i * n)
            n += 1
        if len(str_i) == 9 and len(set(str_i) - {'0'}) == 9:
            if int(str_i) > max_concat:
                 max_concat = int(str_i)
    return max_concat


def problem_039(n=1000):
    """
    Let p be the perimeter of a right angle triangle with integer sides.
    For which value of p < n, is the number of solutions maximised?
    """
    primitive_perimeters = []
    primitive_triangles = []
    for i in range(1, ceil(sqrt(n))):
        for j in range(i + 1, ceil(sqrt(n))):
            if gcd(i, j) == 1 and (i % 2 == 0 or j % 2 == 0):  # By Euclid's formula
                primitive_triangles.append([j*j - i*i, 2*i*j, i*i + j*j])
                primitive_perimeters.append(2*j*j + 2*i*j)
    all_triangles = {}
    for triangle in primitive_perimeters:
        for multiple in range(triangle, n, triangle):
            if multiple in all_triangles:
                all_triangles[multiple] += 1
            else:
                all_triangles[multiple] = 1
    max_count = 0
    max_perimeter = 0
    for perimeter, count in all_triangles.items():
        if count > max_count and perimeter <= n:
            max_perimeter = perimeter
            max_count = count
    return max_perimeter


def problem_040(n=6):
    """
    Champernowne's constant
    If d_n represents the nth digit of the fractional part, find the value of the following expression.
    d_1 x d_10 x ... x d_10**n
    """
    champernowne = "".join([str(i) for i in range(1, 10**n)])
    product = 1
    for i in range(n+1):
        product *= int(champernowne[(10**i)-1])
    return product
        

def problem_041(n=8):
    """
    What is the largest n-digit pandigital prime that exists?  (Eg. 2143)
    """
    # None with 9 or 10 digits because divisible by 3, None n>10 since only 10 digits.
    # Iterate through all i-digit pandigitals, decrementing i to find the least such.
    for i in range(n, 1, -1):
        pandigitals = list(itertools.permutations([str(_) for _ in range(i, 0, -1)]))
        for p in pandigitals:
            int_p = int("".join(p))
            if is_prime(int_p):
                return int_p


def problem_042():
    """
    How many are triangle words?
    """
    triangles = {(i*i - i) // 2 for i in range(100)}
    alpha = "\"abcdefghijklmnopqrstuvwxyz"  # quotes count as zero score
    letter_dict = {alpha[i]: i for i in range(len(alpha))}
    filename = "data/0042_words.txt"
    triangle_words = 0
    with open(filename) as file:
        words = [line.split(",") for line in file]
        for w in words[0]:
            word = str.lower(w)
            word_sum = 0
            for letter in word:
                word_sum += letter_dict[letter]
            if word_sum in triangles:
                triangle_words += 1
    return triangle_words


def problem_043(n=10):
    """
    Find the sum of all 0 to n pandigital numbers with this property.
    """
    total_sum = 0
    pandigitals = list(itertools.permutations([str(_) for _ in range(n)]))
    for p in pandigitals:
        str_p = "".join(p)
        works = True
        primes = [1, 2, 3, 5, 7, 11, 13, 17]
        for triple in range(1, n-2):
            seg = str_p[triple:triple+3]
            if int(seg) % primes[triple] != 0:
                works = False
                break
        if works:
            total_sum += int(str_p)
    return total_sum


def problem_044():
    """
    Find the pair of pentagonal numbers, P_i and P_j, for which their sum and difference are pentagonal and
    the difference is minimised; return that difference.
    """
    pent_list = [i*(3*i - 1) // 2 for i in range(1, 3000)]
    pent_set = set(pent_list)
    smallest_diff = pent_list[-1]
    for i in range(len(pent_list)):
        for j in range(i):
            pent_sum = pent_list[i] + pent_list[j]
            pent_diff = pent_list[i] - pent_list[j]
            if pent_sum in pent_set and pent_diff in pent_set:
                if pent_diff < smallest_diff:
                    smallest_diff = pent_diff
    return smallest_diff


def problem_045():
    """
    Find the next triangle number that is also pentagonal and hexagonal.
    """
    n = 100000
    triangle = set(i * (i + 1) // 2 for i in range(n))
    pentagon = set(i * (3 * i - 1) // 2 for i in range(n))
    hexagon = set(i * (2 * i - 1) for i in range(n))
    intersection = triangle & pentagon & hexagon
    return intersection


def problem_046():
    """
    What is the smallest odd composite that cannot be written as the sum of a prime and twice a square?
    """
    n = 10000
    primes = primeSieve(n)
    prime_set = set(primes)
    odd_comp = set(2*i + 1 for i in range(1, n//2)) - prime_set
    goldbach_sums = set()
    for prime in primes:
        temp_set = set(prime + 2 * i * i for i in range(1, n//100))
        goldbach_sums = goldbach_sums | temp_set
    counter_ex = min(odd_comp - goldbach_sums)
    if counter_ex < n:
        return counter_ex


def problem_047(k=4, N=200000):
    """
    Find the first k consecutive integers to have k distinct prime factors each.
    Return the first of these numbers.
    
    Can I remove the dependence on N?
    """
    # Modified prime sieve to count factors
    factors = [0, 0]
    factors.extend([1 for _ in range(N-1)])
    k_factors = set()
    least_k_streak = 0
    for prime in range(len(factors)):
        if factors[prime] == 1:
            for i in range(2*prime, len(factors), prime):
                factors[i] += 1
        # Find first instance of k-streak
        elif factors[prime] == k + 1:
            k_factors.add(prime)
            streak = True
            for i in range(1, k):
                if prime - i not in k_factors:
                    streak = False
                    break
            if streak:
                least_k_streak = prime - k + 1
                break
    return least_k_streak


def problem_048(n=1000):
    """
    Find the last ten digits of the series, 1^1 + 2^2 + ... + n^n.
    """
    power_sum = 0
    for i in range(1, n+1):
        power_sum += i**i
    return str(power_sum)[-10:]


def problem_049():
    """
    Create an arithmetic sequence of permuted digits, each a prime.
    What 12-digit number do you form by concatenating the three terms in this sequence?
    """
    primes = primeSieve(10000)
    prime_set = set(primes)
    seq_concat = ""
    for i in range(len(primes)):
        if primes[i] < 1000:
            continue
        for j in range(i+1, len(primes)):
            k = 2*primes[j] - primes[i]  # Then this is an arithmetic sequence.
            if k in prime_set:
                i_set = set(d for d in str(primes[i]))
                j_set = set(d for d in str(primes[j]))
                k_set = set(d for d in str(k))
                if len(i_set ^ j_set) == len(i_set ^ k_set) == 0:  # Test if i,j,k are permutations of each other.
                    if primes[i] != 1487:  # Given example
                        seq_concat = str(primes[i]) + str(primes[j]) + str(k)
                        break
    return seq_concat


def problem_050(n=1000000):
    """
    Which prime, below n, can be written as the sum of the most consecutive primes
    """
    primes = primeSieve(n)
    primeSet = set(primes)

    most_primes = 0
    which_prime = 1

    for i in range(len(primes)-1):
        # Only check lengths longer than largest consecutive
        j = most_primes + 1
        prime_sum = sum(primes[i:i+j])
        while prime_sum < n:
            if prime_sum in primeSet:
                most_primes = j
                which_prime = prime_sum
            j += 1
            prime_sum = sum(primes[i:i+j])

    return which_prime


def problem_051(n=8):
    """
    Find the smallest prime which, by replacing part of the number
    (not necessarily adjacent digits) with the same digit, is part of an n prime value family
    """
    primes = primeSieve(1000000)
    primeSet = set(primes)
    max_score = 0
    min_prime = 0
    for p in primes:
        # check if changing digits are in primes
        changes = []
        digit_dict = {d: [] for d in str(p)}
        for d in range(len(str(p))):
            digits = [[d]]

            for s in digit_dict[str(p)[d]]:
                next = [idx for idx in s]
                next.append(d)
                digits.append(next)
            digit_dict[str(p)[d]].extend(digits)
            changes.extend(digits)
        # Rotate through
        for change in changes:
            score = 0
            test = list(str(p))
            min_val = 0
            if 0 in change:
                min_val = 1
            for i in range(min_val, 10):
                for d in change:
                    test[d] = str(i)
                if int("".join(test)) in primeSet:
                    score += 1
            if score > max_score:
                max_score = score
                min_prime = p
                if max_score >= n:
                    return min_prime
    return min_prime


def problem_052(n=6):
    """
    Find the smallest positive integer, x, such that 2x, ... nx, contain the same digits
    """
    i = 0
    searching = True
    while searching:
        i += 1
        digit_set = [0 for _ in range(10)]
        for d in str(i):
            digit_set[int(d)] += 1
        works = True
        for m in range(2, n+1):
            j = m*i
            digit_set_j = [0 for _ in range(10)]
            for d in str(j):
                digit_set_j[int(d)] += 1
            for idx in range(len(digit_set)):
                if digit_set_j[idx] != digit_set[idx]:
                    works = False
        if works:
            searching = False
    return i


def problem_053(n=100):
    """
    How many, not necessarily distinct, values of (k choose r) for 1 <= k <= n, are greater than one-million?
    """
    threshold = 10**6
    count = 0
    # Count twice due to symmetry
    for i in range(n+1):
        for j in range(i//2):
            if comb(i, j) > threshold:
                count += 2
    for i in range(0, n+1, 2):
        if comb(i, i//2) > threshold:
            count += 1
    return count


def problem_054():
    """
    How many poker hands does Player 1 win?
    """
    # Read file
    filename = "data/0054_poker.txt"
    with open(filename) as file:
        poker_rounds = [line.split(" ") for line in file]
    p1_wins = 0
    for round in poker_rounds:
        p1 = round[:5]  # Five cards in each hand
        p2 = round[-5:]
        # Helper function to determine "value" of each hand
        p1_score = poker_hand(p1)
        p2_score = poker_hand(p2)
        print(p1_score, p2_score)
        # Compare results from each hand
        if p1_score > p2_score:
            p1_wins += 1
    return p1_wins


def poker_hand(cards):
    # Determine card values
    values = {"23456789TJQKA"[i]: i for i in range(13)}
    nums = [0 for _ in range(13)]
    for card in cards:
        nums[values[card[0]]] += 1

    # Encode score [xx][xx]: First set win style, second tie-breaker
    # Score defaults to High Card
    score = max([values[card[0]] for card in cards]) + 2

    # Find pairs
    pairs = []
    triple = []
    four = []
    for i in range(len(nums)):
        if nums[i] == 2:
            pairs.append(i+2)
            score = i + 2
        elif nums[i] == 3:
            triple.append(i+2)
            score = i + 2
        elif nums[4] == 4:
            four.append(i+2)
            score = i + 2
    
    # Find straights
    straight = "11111" in  "".join([str(i) for i in nums])

    # Find flushes
    suits = [card[1] for card in cards]
    flush = len(set(suits)) == 1

    if flush:
        score += 400
    if straight:
        score += 500
    if len(four) > 0:
        score += 700
    if len(triple) > 0 and len(pairs) > 0:
        score += 600
    if len(triple) > 0:
        score += 300
    if len(pairs) == 2:
        score += 200
    if len(pairs) == 1:
        score += 100
    return score


def problem_055(n=10000):
    """
    How many Lychrel numbers are there below n? (over fifty iterations)
    """
    count = 0
    for num in range(n+1):
        i = 0
        test_val = num
        while True:  # Test if palindrome
            if i > 50:
                count += 1
                break
            reverse = reversed(str(test_val))
            test_val += int("".join(reverse))
            if isPalindrome(test_val):
                break
            i += 1
    return count


def problem_056(n=100):
    """
    Considering natural numbers of the form, a**b, where a, b < n, what is the maximum digital sum?
    """
    max_digit_sum = 0
    for a in range(1, n):
        for b in range(1, n):
            num = a**b
            digit_sum = sum([int(i) for i in str(num)])
            if digit_sum > max_digit_sum:
                max_digit_sum = digit_sum
    return max_digit_sum


def problem_058(p=1, q=10):
    """
    How large must an Ulam sprial be for the percentage of 
    Primes along the diagonals to fall below p/q?
    """
    primes = 0
    total = 1
    layer = 0
    while primes * q >= p * total or total == 1:
        layer += 1
        total += 4
        # Non-square values along the diagonals
        a = (4 * layer * layer) - (2 * layer) + 1
        b = (4 * layer * layer) + 1
        c = (4 * layer * layer) + (2 * layer) + 1
        if is_prime(a):  # Miller-Rabin Primality Test
            primes += 1
        if is_prime(b):
            primes += 1
        if is_prime(c):
            primes += 1
    side_length = 2 * layer + 1
    return side_length


def wip_problem_059():
    """
    XOR cipher
    """
    filename = "data/0059_cipher.txt"
    message = []
    with open(filename) as file:
        words = [line.split(",") for line in file]
        for i in range(len(words[0])):
            message.append(int(words[0][i]))

    # iterate through three lowercase characters


def problem_060(n=5):
    """
    Find the lowest sum for a set of n primes for which any two primes concatenate to produce another prime.
    """
    G = nx.Graph()
    primes = primeSieve(10000)  # Can I remove this dependence on N?
    for p1 in range(1, len(primes)):
        for p2 in range(p1):
            concat1 = primes[p1] * 10 ** len(str(primes[p2])) + primes[p2]
            concat2 = primes[p1] + primes[p2] * 10 ** len(str(primes[p1]))
            if is_prime(concat1) and is_prime(concat2):
                G.add_edge(primes[p1], primes[p2])
    n_clique_sum = [sum(c) for c in nx.find_cliques(G) if len(c) == n]
    least_sum = min(n_clique_sum)
    return least_sum

"""
for problem in [problem_060]:
    t = time.process_time()
    problem()
    time_elapsed = time.process_time() - t
    print(time_elapsed)
"""

def problem_063():
    """
    How many positive n-digit numbers are nth powers?
    """
    # We can limit n <= 9
    count = 9  # All single digits
    for i in range(2, 10):
        for j in range(2, 50):
            if len(str(i**j)) == j:
                count += 1
    return count


def problem_064(n=10000):
    """
    How many continued fractions for sqrt(i), i < n, have an odd period?
    """
    odd_count = 0
    for i in range(1, n+1):
        if floor(sqrt(i))**2 == i:
            continue
        whole = floor(sqrt(i))
        cont_frac = []
        m = 0
        d = 1
        a = whole
        while a != 2*whole:  # https://en.wikipedia.org/wiki/Periodic_continued_fraction#Canonical_form_and_repetend
            m = d*a - m
            d = (i - m**2) // d
            a = floor((sqrt(i) + m) // d)
            cont_frac.append(a)
        if len(cont_frac) % 2 == 1:
            odd_count += 1
    return odd_count


def problem_065(n=100):
    """
    Find the sum of digits in the numerator of the n-th convergent of the continued fraction for e.
    """
    # Build continued fraction representation
    val = 2
    e_cont_frac = [2]
    for _ in range(n//3 + 1):
        e_cont_frac.extend([1, val, 1])
        val += 2
    # Cascade up the continued fraction
    numerator = 1
    denominator = e_cont_frac[n-1]
    for i in range(n-2, -1, -1):
        new_numerator = e_cont_frac[i] * denominator + numerator
        numerator = denominator
        denominator = new_numerator
    # Undo the final switch
    numerator, denominator = denominator, numerator
    return sum([int(i) for i in str(numerator)])


def problem_067():
    """
    Find the maximum total from top to bottom of the triangle below
    """
    filename = "data/0067_triangle.txt"
    max_val = 0
    with open(filename) as file:
        pyramid = [line.split(" ") for line in file]
        for line in range(len(pyramid)):
            for item in range(len(pyramid[line])):
                pyramid[line][item] = int(pyramid[line][item])
        for row in range(len(pyramid)-2, -1, -1):
            for item in range(len(pyramid[row])):
                pyramid[row][item] = int(pyramid[row][item])
                pyramid[row][item] += max(pyramid[row+1][item], pyramid[row+1][item+1])
        max_val = pyramid[0][0]
    return max_val


def problem_069(n=10**6):
    """
    Find the value of i < n for which n/phi(i) is a maximum.
    """
    # phi(n) is minimized at primorial values
    phi = 1
    primes = primeSieve(ceil(sqrt(n)))
    num = 1
    i = 0
    # so n/ phi(n) is maximized at the largest primorial < n.
    while num*primes[i] < n:
        num *= primes[i]
        phi *= primes[i]-1
        i += 1
    return num


def problem_075(n=1500000):
    """
    Given that L is the length of the wire, for how many values of
    L <= n can exactly one integer sided right angle triangle be formed?
    """
    primitive_perimeters = []
    primitive_triangles = []
    for i in range(1, ceil(sqrt(n))):
        for j in range(i + 1, ceil(sqrt(n))):
            if gcd(i, j) == 1 and (i % 2 == 0 or j % 2 == 0):  # By Euclid's formula
                primitive_triangles.append([j*j - i*i, 2*i*j, i*i + j*j])
                primitive_perimeters.append(2*j*j + 2*i*j)
    all_triangles = {}
    for triangle in primitive_perimeters:
        for multiple in range(triangle, n, triangle):
            if multiple in all_triangles:
                all_triangles[multiple] += 1
            else:
                all_triangles[multiple] = 1
    unique_count = 0
    for perimeter, count in all_triangles.items():
        if count == 1 and perimeter <= n:
            unique_count += 1

    return unique_count


def problem_081():
    """
    Find the minimal path sum from the top left to the bottom right by only moving right and down
    """
    filename = "data/0081_matrix.txt"
    min_val = 0
    with open(filename) as file:
        matrix = [line.split(",") for line in file]
        for line in range(len(matrix)):
            for item in range(len(matrix)):
                matrix[line][item] = int(matrix[line][item])
                if line == 0 and item != 0:
                    matrix[line][item] += matrix[line][item-1]
                if line != 0 and item == 0:
                    matrix[line][item] += matrix[line-1][item]
        for row in range(1, len(matrix)):
            for item in range(1, len(matrix[row])):
                # matrix[row][item] = int(matrix[row][item])
                matrix[row][item] += min(matrix[row-1][item], matrix[row][item-1])
        min_val = matrix[-1][-1]
    return min_val


def problem_089():
    """
    Find the number of characters saved by writing each roman numeral in its minimal form.
    """
    filename = "data/0089_roman.txt"
    with open(filename) as file:
        romans = [line.strip() for line in file]
    chars_removed = 0
    for roman in romans:
        # Resolve 1's place: IIII -> IV or VIIII -> IX
        if "VIIII" in roman:
            chars_removed += 3
        elif "IIII" in roman:
            chars_removed += 2
        # Resolve 10's place: XXXX -> XL or LXXXX -> XC
        if "LXXXX" in roman:
            chars_removed += 3
        elif "XXXX" in roman:
            chars_removed += 2
        # Resolve 100's place: CCCC -> CD or DCCCC -> CM
        if "DCCCC" in roman:
            chars_removed += 3
        elif "CCCC" in roman:
            chars_removed += 2
    return chars_removed


def problem_092(n=10**7):
    """
    The next term in a sequence is the sum of the digits squared.
    How many below n lead to 89? (All starting values end in 1 or 89)
    """
    count = 0
    n_dict = {1: False, 89: True}  # Use Union find-esque algorithm. All numbers which lead to 89 are True
    digit_dict = {str(i): i*i for i in range(10)}
    for i in range(1, n+1):
        x_list = [i]
        while True:
            x_list.append(sum([digit_dict[d] for d in str(x_list[-1])]))
            if x_list[-1] in n_dict:
                to_89 = n_dict[x_list[-1]]
                count += int(to_89)
                for x in x_list:
                    n_dict[x] = to_89
                break
    return count


def problem_096():
    """
    Solve a sudoku puzzle
    """
    # Read file into list of sudoku puzzles
    filename = "data/0096_sudoku.txt"
    puzzles = []
    with open(filename) as file:
        current = []
        i = 0
        for line in file:
            if line[0] == "G" or line[0] == '\n':
                continue
            else:
                i += 1
                nums = line[:9]
                current.append([int(num) for num in nums])
                if i % 9 == 0:
                    puzzles.append(current)
                    current = []
    # SOLVE EACH PUZZLE
    i = 0
    digit_sum = 0
    for puzzle in puzzles:
        complete, solved = solveSudoku(puzzle)
        puzzle_sum = solved[0][0] * 100 + solved[0][1] * 10 + solved[0][2]
        digit_sum += puzzle_sum
        i += 1
    return digit_sum


def solveSudoku(puzzle):
    # Puzzle is a 9 by 9 array, where empty spaces are marked as 0
    SIZE = len(puzzle)
    SQRT = int(sqrt(SIZE))
    incomplete = True

    while incomplete:  # Attempt to solve
        unchanged = True
        complete = 0

        # What is used in each row, column, box
        rows = [{puzzle[i][j] for j in range(SIZE)} - {0} for i in range(SIZE)]
        cols = [{puzzle[j][i] for j in range(SIZE)} - {0} for i in range(SIZE)]
        boxs = [{puzzle[SQRT*(i//SQRT) + (j//SQRT)][SQRT*(i % SQRT) + (j % SQRT)] for j in range(SIZE)} - {0} for i in range(SIZE)]

        # List all possible options for each location
        possible = [[set(i for i in range(1, SIZE + 1)) for col in range(SIZE)] for row in range(SIZE)]
        for row in range(SIZE):
            for col in range(SIZE):
                if puzzle[row][col] == 0:
                    possible[row][col] -= rows[row]
                    possible[row][col] -= cols[col]
                    possible[row][col] -= boxs[3*(row//3) + (col//3)]
                else:
                    possible[row][col] = set()
                    complete += 1
                # Only one option, so put it in solution
                if len(possible[row][col]) == 1:
                    val = list(possible[row][col])
                    puzzle[row][col] = val[0]
                    complete += 1
                    unchanged = False
                # Determine if any numbers unique to row, column, box
        for row in range(SIZE):
            for col in range(SIZE):
                if len(possible[row][col]) > 1:  # Duplicate possible object for each row, col, box
                    unique_row = {i for i in possible[row][col]}
                    unique_col = {i for i in possible[row][col]}
                    unique_box = {i for i in possible[row][col]}
                    for i in range(SIZE):
                        if i != row:
                            unique_col -= possible[i][col]
                        if i != SQRT*(row % SQRT) + (col % SQRT):
                            unique_box -= possible[SQRT*(row//SQRT) + (i//SQRT)][SQRT*(col//SQRT) + (i % SQRT)]
                        if i != col:
                            unique_row -= possible[row][i]

                    if len(unique_col) == 1:
                        val = list(unique_col)
                        puzzle[row][col] = val[0]
                        complete += 1
                        unchanged = False
                        break
                    if len(unique_box) == 1:
                        val = list(unique_box)
                        puzzle[row][col] = val[0]
                        complete += 1
                        unchanged = False
                        break
                    if len(unique_row) == 1:
                        val = list(unique_row)
                        puzzle[row][col] = val[0]
                        complete += 1
                        unchanged = False
                        break

        # Attempt to solve by contradiction
        if unchanged:
            for row in range(SIZE):
                for col in range(SIZE):
                    if len(possible[row][col]) > 1:
                        copy = [[puzzle[i][j] for j in range(SIZE)] for i in range(SIZE)]
                        for val in possible[row][col]:
                            copy[row][col] = val
                            solved, copy = solveSudoku(copy)
                            if solved:
                                return solved, copy
                return False, puzzle

        if complete >= SIZE * SIZE:
            done = True
            # Check sum along row and column
            for i in range(SIZE):
                col_sum = 0
                if sum(puzzle[i]) != 45:
                    done = False
                for j in range(SIZE):
                    col_sum += puzzle[j][i]
                if col_sum != 45:
                    done = False
            return done, puzzle
    return False, puzzle


def problem_097(n=10):
    """
    Find the last n digits of the largest non-Mersenne prime
    """
    num = 2**7830457 % 10**n
    num *= 28433
    num = (num + 1) % 10**n
    return num


def problem_100(n=10**12, p=1, q=2):
    """
    Find the first arrangement to contain over n balls in total, such that
    choosing two without replacement gives p/q probability of the dominant color.
    Return how many of the dominant color. GENERALIZED TO ALL FRACTIONS
    """
    dominant = 1
    total = 1
    g = gcd(q-1, 2)
    slope_approx = [[(q-1)//g, 2//g]]  # Fractional approximations to 1 - sqrt(q)
    i = 0
    while total < n:  # Vieta jumping through positive hyperbolic solutions
        a = slope_approx[-1][0]
        b = slope_approx[-1][1]
        g = gcd((q-1)*b, a + 2*b)
        slope_approx.append([((q-1)*b)//g, (a + 2*b)//g])
        if i % 2 == 0:
            dominant += b
            total += a + b
        i += 1
    return slope_approx[-1][0]


def problem_101(n=11):
    """
    Find the sum of first incorrect terms (FIT)  for the optimal polynomials upto degree n.
    """
    def f(x):
        return 1 - x + x**2 - x**3 + x**4 - x**5 + x**6 - x**7 + x**8 - x**9 + x**10
    terms = [f(i) for i in range(1, n+1)]
    wrong = []
    for term in range(1, len(terms)):
        difference = [terms[:term]]
        while len(difference[-1]) > 1:
            dif = [difference[-1][i] - difference[-1][i-1] for i in range(1, len(difference[-1]))]
            difference.append(dif)
        for i in range(len(difference)-1, -1, -1):
            if i == len(difference)-1:
                dif = difference[i][0]
                difference[i].append(dif)
            else:
                dif = difference[i][-1] + difference[i+1][-1]
                difference[i].append(dif)
        wrong.append(difference[0][-1])
    return sum(wrong)


def wip_problem_104():
    """
    Given that F_k is the first Fibonacci number for which the first nine digits AND the last nine digits
    are 1-9 pandigital, find k.
    """
    fib_a = 0
    fib_b = 1
    zero = {'0'}
    i = 2
    while True:
        fib_a, fib_b = fib_b, fib_a + fib_b
        r = mpz(fib_b)
        str_fib = r.digits()
        front = set(str_fib[:9])
        back = set(str_fib[-9:])
        if len(front) == 9 and len(front & zero) == 0:
            if len(back) == 9 and len(back & zero) == 0:
                break
        i += 1
    return i


def wip_problem_111(n):
    """
    Given a integer length, n, determine the maximum number of each digit, 0-9, while still a prime.
    Return the sum of the primes with these maximal digit repeats.
    """
    # For digit in range(0, 10):
    # work through how many digits could appear
    # if that digits sum is not zero, add to the total
    # iterate through different choose functions where the non-digit values are
    totalSum = 0
    for digit in range(0, 10):
        num = digit * n
    print()


def problem_112(p=99, q=100):
    """
    We shall call a positive integer whose digits are neither increasing nor decreasing a "bouncy" number.
    Find the least number for which the proportion of bouncy numbers is exactly p/q.
    """
    bouncy_count = 0
    i = 1
    while bouncy_count * q < p * i:
        i += 1
        str_i = mpz(i).digits()
        val = int(str_i[0])
        # Check if number is increasing or decreasing
        increasing = True
        decreasing = True
        for c in str_i:
            if int(c) < val:
                increasing = False
            if int(c) > val:
                decreasing = False
            val = int(c)
            if not increasing and not decreasing:
                bouncy_count += 1
                break
    return i


def problem_121(n=15):
    # n is the number of turns
    """
    A bag initially contains one red disc and one blue disc.
    A disc is randomly chosen then returned to the bag. After, an extra red disc is added.
    The player wins if they have taken more blue discs than red discs at the end of the game.
    Find the maximum prize fund that should be allocated to a single game in which fifteen turns are played.
    """
    initial_balls = 2
    trials = n
    denominator = factorial(n+1)
    prob = recursion_121(trials + 1, initial_balls, 0, 1)

    prize = (denominator // prob)
    return prize


def recursion_121(max_balls, balls, blue_count, numerator):
    if balls > max_balls + 1:  # Too many trials, loss
        return 0
    if blue_count >= ceil(max_balls/2):  # Majority blue, win
        probability = numerator * factorial(max_balls) // factorial(balls - 1)
        return probability
    if balls - blue_count > ceil((max_balls + 1)/2):  # Majority red, discard
        return 0
    # Another draw, add red ball
    probability = recursion_121(max_balls, balls+1, blue_count+1, numerator)  # Draw blue
    probability += recursion_121(max_balls, balls+1, blue_count, numerator * (balls-1))  # Draw red
    return probability


def problem_169(n=10**25):
    """
    Return the number of partitions of n using only integer powers of 2, each no more than twice.
    """
    binary = bin(n)[2:]
    ones = []
    zero_gaps = 0
    zero = True
    for i in range(1, len(binary)+1):
        if binary[-i] == '1':
            ones.append(i)
            zero = True
        elif zero:
            zero = False
            zero_gaps += 1
    soln = [1, 1]
    for i in range(len(ones)):
        gap = 1
        if i == 0:
            gap *= ones[i]
        else:
            gap *= ones[i] - ones[i-1]
        soln.append(gap*soln[-1] + (soln[-1]-soln[-2]))
    return soln[-1]


def problem_243(p=15499, q=94744):
    phi = 1
    primes = primeSieve(q)
    num = 2
    i = 1
    # n/ phi(n) is maximized at the largest primorial < n
    while phi / num > p / q:
        num *= primes[i]
        phi *= primes[i]-1
        i += 1
    # Correct for the minus 1 in the denominator
    if phi / (num - 1) > p / q:
        for i in range(2, primes[i]):
            if phi * i / (i * num - 1) < p / q:
                phi *= i
                num *= i
                break
    return num


def wip_problem_500(n=500500):
    """
    Find the smallest number with 2**n factors
    """
    MAX = 8000000
    primes = primeSieve(MAX)
    MOD = 500500507
    prime_powers = []
    for prime in primes:
        i = prime
        while i < MAX:
            prime_powers.append(i)
            i *= i
    prime_powers.sort()
    x = 1
    print(prime_powers[:10])
    for p in range(n):
        x = (x * prime_powers[p]) % MOD
    print(len(prime_powers))
    print(x)
    return x


def wip_problem_700():
    """
    Consider x = an % b. x is in X if smaller than all previous x. Find the sum of all X.
    """
    ADD = 1504170715041707
    MOD = 4503599627370517

    Eulercoins = []
    a = MOD
    b = ADD
    while a != b:
        t = b
        b = b - (a % b)
        a = t
        Eulercoins.append(a)
    sum_euler = sum(Eulercoins)
    print(sum_euler)
    return sum_euler


def wip_problem_719(n=10**6):
    """
    We define an S-number to be a perfect square and its square root can be obtained
     by splitting the decimal representation into 2 or more numbers then adding the numbers.
    Find the sum of all S numbers from 1 to n^2.
    """
    s_sum = 0
    for i in range(4, n + 1):
        str_sq = str(i * i)
        length = len(str_sq) - 1
        str_b = '0' + str(length) + 'b'
        for j in range(2**length):
            bin_j = format(j, str_b)
            sum_v = 0
            temp = str_sq[0]
            for k in range(length):
                if bin_j[k] == '0':
                    temp += str_sq[k+1]
                else:
                    sum_v += int(temp)
                    temp = str_sq[k+1]
                    if sum_v > i:
                        break
            sum_v += int(temp)
            if sum_v == i:
                s_sum += i * i
                break
        if i % 10000 == 0:
            print(i)
    print(s_sum)


def wip_problem_719_a(n=10**6):
    """
    We define an S-number to be a perfect square and its square root can be obtained
     by splitting the decimal representation into 2 or more numbers then adding the numbers.
    Find the sum of all S numbers from 1 to n^2.
    """
    s_sum = 0
    for i in range(4, n + 1):
        str_sq = str(i * i)
        length = len(str_sq)
        if recursion_719(i, 1, length):
            s_sum += i * i
        if i % 10000 == 0:
            print(i)
    print(s_sum)


def recursion_719(num, length, max_len, sum_s=0, temp=0):
    if length > max_len:
        return False
    current = int(str(num*num)[temp:length])
    if current + sum_s > num:
        return False
    if length == max_len and sum_s + current == num:
        return True
    # Case 0, extend
    if recursion_719(num, length+1, max_len, sum_s, temp):
        return True
    # Case 1, add
    if recursion_719(num, length+1, max_len, sum_s + current, length):
        return True
    return False


def problem_751(n=24):
    """
    Find a convergent value for a specific function, so that f(x) = x
    """
    theta = 2
    for j in range(50):  # Repeat until convergence
        theta = float(theta)
        b = [theta]
        a = [floor(theta)]
        for i in range(24):  # Lengthen the value
            b_n = a[-1] * (b[-1] - a[-1] + 1)
            b.append(b_n)
            a.append(floor(b_n))
        theta = str(a[0]) + "." + "".join([str(i) for i in a[1:]])
    return theta


def wip_problem_808(n=50):
    """
    Return the sum of the first n reversible prime squares
    """
    primes = primeSieve(50000000)
    prime_squares = [prime * prime for prime in primes]
    reversible = []
    prime_square_set = set(prime_squares)
    for ps in prime_squares:
        s = str(ps)
        rs = [s[-i] for i in range(1, len(s)+1)]
        val = int("".join(rs))
        if val in prime_square_set:
            if val != ps:
                reversible.append(ps)
    reversible_sum = sum(reversible[:n])
    return reversible_sum


def problem_816(n=2000000):
    """
    s_0 = 290797
    s_n+1 = s_n**2 mod 50515093
    P_n = (s_2n, 2_2n+1). Find the minimum distance between any two points.
    Give your answer rounded to 9 places after the decimal point.
    """
    # STRATEGY: Divide the points into sqrt(n) bins along the x and y.
    # Compare points within each bin (and each neighboring bin)
    MOD = 50515093
    DIV = 100000
    s1 = 290797
    s2 = (s1 * s1) % MOD
    boxs = [[[] for __ in range(MOD//DIV + 1)] for _ in range(MOD//DIV + 1)]
    for i in range(n):
        p = [s1, s2]
        boxs[s1//DIV][s2//DIV].append(p)
        s1 = (s2 * s2) % MOD
        s2 = (s1 * s1) % MOD
    # Find the distances between points in the same sectors
    min_dist = MOD * MOD
    for i in range(len(boxs)):
        for j in range(len(boxs)):
            for p_i in range(len(boxs[i][j])):
                for p_j in range(len(boxs[i][j])):
                    if p_i == p_j:
                        continue
                    p1 = boxs[i][j][p_i]
                    p2 = boxs[i][j][p_j]
                    x = p1[0] - p2[0]
                    y = p1[1] - p2[1]
                    if x*x + y*y < min_dist:
                        # print("NEW MIN", x*x + y*y)
                        min_dist = x*x + y*y
    return sqrt(min_dist)


def wip_problem_837(m, n, mod=100):
    """
    Let a(m, n) be the number of different three-object Amidakujis (Group S3)
    that have m rungs (transpositions) between A and B, and n rungs (transpositions)
    between B and C, and whose outcome is the identity permutation.
    Give your answer modulo [mod]
    """
    total = 0
    if m % 2 == n % 2:  # m, n must have same parity
        if m % 2 == 0:  # even case
            for i in range(0, min(m, n), 6):
                m_i = m - i
                n_i = n - i
                total += comb(i, m)

        else:  # odd case
            for i in range(0, min(m, n), 6):
                print(i)
    else:
        return total


