import scipy.special

def binomial(n, k):
    ret = 1
    for i in range(0, k):
        ret *= n-i
        ret /= i+1
    return ret


def sign_test(target, predictionsA, predictionsB):
    plus = 0
    minus = 0
    null = 0

    A_false_pos = 0
    A_false_neg = 0
    B_false_pos = 0
    B_false_neg = 0

    for i in range(len(target)):
        if (predictionsA[i] == predictionsB[i]):
            null += 1
        elif (predictionsA[i] == target[i]):
            plus += 1
            if (predictionsB[i] == 0):
                B_false_pos += 1
            else:
                B_false_neg += 1
        else:
            minus += 1
            if (predictionsA[i] == 0):
                A_false_pos += 1
            else:
                A_false_neg += 1

    print("\n\nSign test results: ")
    print("Plus: ", plus, " Minus:", minus, " Null:", null)
    print("A false pos:", A_false_pos, " false neg:", A_false_neg)
    print("B false pos:", B_false_pos, " false neg:", B_false_neg)
    n = 2 * ((null + 1) // 2) + plus + minus
    k = (null + 1) // 2 + min(plus, minus)
    p = 0

    for i in range(k + 1):
        p += scipy.special.comb(n, i, exact=True)

    return p / 2 ** (n - 1)
