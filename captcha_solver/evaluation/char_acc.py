def longest_common_subsequence_length(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

# character_accuracy for a single CAPTCHA
def character_accuracy(gt: str, pred: str) -> float:
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    lcs_len = longest_common_subsequence_length(gt, pred)
    return lcs_len / len(gt)

# example usage
if __name__ == "__main__":
    ground_truth = "abcde"
    pred = "abbcde"
    print(longest_common_subsequence_length(ground_truth, pred))
