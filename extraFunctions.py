from math import sqrt, floor, ceil
from time import time
import random


def primeSieve(n):
    z = [False, False]
    z.extend([True for _ in range(n-1)])
    for prime in range(ceil(sqrt(len(z)))):
        if z[prime]:
            for i in range(prime*prime, len(z), prime):
                z[i] = False
    primes = [i for i in range(len(z)) if z[i]]
    return primes


def EuclideanAlg(a, b):
    if b > a:
        return EuclideanAlg(b, a)
    if b == 0:
        return a
    else:
        return EuclideanAlg(b, a % b)


def timeTest(f, n):
    start = time()
    f(n)
    finish = time() - start
    print(n, finish)


def isPalindrome(s):
    S = str(s)
    palindrome = True
    for i in range(len(S)//2):
        if S[i] != S[-i-1]:
            palindrome = False
    return palindrome


def divisor_count(n, primes):
    exponents = []
    p_index = 0
    while n != 1:
        p = primes[p_index]
        exponents.append(1)
        while n % p == 0:
            n = n//p
            exponents[-1] += 1
        p_index += 1
    divisors = 1
    for d in exponents:
        divisors *= d
    return divisors


def totient_a(n):
    z = [False, False]
    phi = {i: 1 for i in range(n+1)}
    z.extend([True for _ in range(n-1)])
    for prime in range(ceil(sqrt(len(z)))):
        if z[prime]:
            phi[prime] = prime - 1
            for i in range(prime*prime, len(z), prime):
                z[i] = False
                phi[i] *= phi[prime] * phi[i//prime]
    primes = [i for i in range(len(z)) if z[i]]
    return primes


def totient_b(n):
    primes = primeSieve(n)
    totient_dict = {i: i-1 for i in primes}
    for i in range(4, n+1):
        temp = 1
        prime = 0
        while i not in totient_dict:
            if prime == 0:
                if i % 4 == 0:
                    temp *= 2
            if i % primes[prime] == 0:
                temp *= totient_dict[primes[prime]]


def is_prime(p, k=40):
    """
    Probabilistic method returns whether p is prime or not. Uses Miller-Rabin test.
    Conducts k tests, with 40 being optimal
    """
    # Composite if even
    if p == 2 or p == 3:
        return True
    elif p % 2 == 0:
        return False
    elif p == 1:
        return False
    # Miller-Rabin
    r, s = 0, p - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randrange(2, p - 1)
        x = pow(a, s, p)
        if x == 1 or x == p - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, p)
            if x == p - 1:
                break
        else:
            return False
    return True


def is_pandigital(s, digits="123456789"):
    return set(s) == set(digits)


def wieferich(n):
    primes = primeSieve(n)
    w = []
    for p in primes:
        if (8**(p-1) - 1) % (p * p) == 0:
            print(p)
            w.append(p)
    print(w)


# wieferich(100000)

"""
n = 10000
timeTest(totient_a, n)
timeTest(totient_b, n)
"""

