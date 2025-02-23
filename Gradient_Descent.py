class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init  # Initial guess
        for _ in range(iterations):
            gradient = 2 * x  # Derivative of f(x) = x^2 is f'(x) = 2x
            x = x - learning_rate * gradient  # Gradient Descent update step
        
        return round(x, 5)

# Example usage
solution = Solution()
print(solution.get_minimizer(0, 0.01, 5))  
print(solution.get_minimizer(10, 0.01, 5)) 
