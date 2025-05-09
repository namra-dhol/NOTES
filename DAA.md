
---

### 🧠 **DAA UNIT-1 – Basics of Algorithms**

#### 📌 **1. What is an Algorithm?**

* Step-by-step procedure to solve a problem.
* Must be: **Finite, Clear, Effective**.

#### ✅ **2. Algorithm Characteristics**

* Input
* Output
* Definiteness (clear steps)
* Finiteness (ends after steps)
* Effectiveness (simple, doable steps)

#### 📊 **3. Performance Analysis**

* **Space Complexity**: Memory used
* **Time Complexity**: Time taken

#### ⏱️ **4. Asymptotic Notations**

* **Big-O (O)** – Worst case
* **Omega (Ω)** – Best case
* **Theta (Θ)** – Average case / tight bound

📌 Examples:

* `O(n)`: linear
* `O(n^2)`: quadratic
* `O(log n)`: logarithmic

#### 📈 **5. Growth of Functions**

* Used to compare algorithms.
* Focuses on how input size affects time.

#### 🧮 **6. Recurrence Relations**

* Equations that define time complexity of recursive algorithms.
* Solve using:

  * Substitution method
  * Recursion tree
  * Master’s Theorem

Example:
T(n) = 2T(n/2) + n → **O(n log n)**

#### 🔍 **7. Algorithm Analysis Types**

* **Worst Case**: Max time
* **Best Case**: Min time
* **Average Case**: Expected time

---

### 📝 Quick Tips to Remember

* **FIRE** rule for Algorithms: **F**inite, **I**nput/Output, **R**eal steps, **E**nd.
* **Big-O** = "Oh no!" = worst case
* Think **"log n" is better than "n", and "n^2" is slower**

---


---

## 🧠 DAA UNIT-2 – Divide & Conquer (D\&C)

### 📌 1. **Divide and Conquer Concept**

* Break big problem into smaller parts.
* Solve each part **recursively**.
* Combine the results.

👉 Example: Merge Sort, Quick Sort, Binary Search, Strassen’s Matrix Multiplication.

---

### 🔄 2. **Recurrence Relations**

* Time complexity of recursive algorithms.
* Written as:
  `T(n) = aT(n/b) + f(n)`
  where:

  * `a`: no. of subproblems
  * `b`: size reduction
  * `f(n)`: cost of divide/combine

---

### 🧩 3. **Methods to Solve Recurrence**

#### ✅ Substitution Method:

* Guess the solution, prove by induction.

#### 🌳 Recursion Tree:

* Draw tree of recursive calls, add cost level by level.

#### 📐 Master’s Theorem:

For `T(n) = aT(n/b) + f(n)`:

* Compare `f(n)` with `n^log_b(a)` to find time.

---

### 🔍 4. **Binary Search**

* Repeatedly divide sorted array in half.
* Time complexity: **O(log n)**
* Types: Iterative, Recursive

---

### 🔢 5. **Multiplying Large Integers**

* Divide numbers, multiply smaller parts, combine.
* D\&C speeds up multiplication.

---

### 🧮 6. **Matrix Multiplication (Strassen’s Algorithm)**

* Faster than standard method.
* Reduces number of multiplications from 8 to 7.

---

### 🧹 7. **Merge Sort**

* **Divide** array → **Sort** parts → **Merge** them
* Time complexity: **O(n log n)**

---

### ⚡ 8. **Quick Sort**

* Choose a **pivot**.
* Elements < pivot → left, > pivot → right.
* Recursively sort both sides.
* Time:

  * **Best/Average**: O(n log n)
  * **Worst**: O(n²) if bad pivot

---

## ✍️ Easy Memory Tricks:

| Topic          | Easy Tip                                           |
| -------------- | -------------------------------------------------- |
| D\&C           | Divide, Solve, Combine                             |
| Binary Search  | Always cut the list in half                        |
| Master Theorem | Use when recurrence looks like `T(n)=aT(n/b)+f(n)` |
| Merge Sort     | Always safe O(n log n)                             |
| Quick Sort     | Fast but risky if pivot is bad                     |
| Strassen       | Matrix magic – 7 multiplications                   |

---


---

## 🧠 DAA UNIT-3 – Greedy Algorithm

### 💡 What is a Greedy Algorithm?

* Solves problems **step-by-step**.
* Always picks the **best option at the moment** (locally optimal).
* Aims to find **global optimum** (but not always guaranteed).

---

### ✅ Characteristics of Greedy Algorithm

1. **Candidate set** (choices)
2. **Selection function** – picks the best choice
3. **Feasibility function** – checks if it’s valid
4. **Solution function** – checks if goal is reached

---

### 💰 1. **Make Change Problem**

* Choose highest denomination coin ≤ remaining amount.
* Fast, but may not always give minimum coins in non-standard coin sets.

---

### 🌳 2. **Minimum Spanning Tree (MST)**

#### 📌 Kruskal’s Algorithm:

* Sort edges by weight
* Pick smallest edge that doesn’t form a cycle
* Use **Disjoint Set (Union-Find)**

#### 📌 Prim’s Algorithm:

* Start from one node
* Grow tree by adding shortest edge to a new node

🕸️ Use to connect all nodes with **minimum total edge weight**.

---

### 🚀 3. **Dijkstra’s Algorithm**

* Finds **shortest path** from one node to all others.
* Always chooses **nearest unvisited node** next.
* Works only for **non-negative edge weights**.

---

### 🎒 4. **Fractional Knapsack Problem**

* Sort by **value/weight ratio**
* Take full items until full; take a fraction if needed.
* Can split items.

---

### 🎯 5. **Activity Selection Problem**

* Goal: Max number of non-overlapping activities.
* Sort by **end time**, pick the next activity that starts after the last selected one.

---

### 🧑‍💻 6. **Job Scheduling with Deadlines**

* Each job has **profit** and **deadline**.
* Pick job with **highest profit** and schedule it as late as possible before its deadline.

---

### 🔡 7. **Huffman Coding**

* Compresses data using **variable-length prefix codes**.
* More frequent chars get **shorter codes**.
* Build a binary tree based on frequency.

---

## ✍️ Quick Memory Tips

| Problem                  | Greedy Strategy                            |
| ------------------------ | ------------------------------------------ |
| Make Change              | Pick biggest coin that fits                |
| Kruskal’s MST            | Pick lightest edge without forming a cycle |
| Prim’s MST               | Grow MST from current node                 |
| Dijkstra’s Shortest Path | Visit closest node next                    |
| Fractional Knapsack      | Max value/weight ratio                     |
| Activity Selection       | Sort by finish time                        |
| Job Scheduling           | Max profit job before deadline             |
| Huffman Coding           | Merge least frequent symbols first         |

---



## 🧠 DAA UNIT-4 – Dynamic Programming (DP)

### 💡 What is Dynamic Programming?

* Solves problems by **breaking them into sub-problems**
* Stores results to **avoid repeating** calculations (memoization)
* Used for **optimization problems** (maximum, minimum, shortest path, etc.)

---

### 📌 Key Steps in DP

1. **Characterize** optimal solution
2. **Recursively define** solution
3. **Compute** in bottom-up table
4. **Construct** actual solution (if needed)

---

### 🌟 Principle of Optimality

* If overall solution is optimal, then **every sub-part must also be optimal**
* Without this property, **DP won't work**

---

## ✨ Important Problems in DP

### 🧮 1. **Binomial Coefficient**

* Count combinations:
  `C(n, k) = C(n-1, k-1) + C(n-1, k)`
* Base cases:
  `C(n, 0) = C(n, n) = 1`

---

### 💰 2. **Make Change Problem**

* Goal: Make amount using **minimum coins**
* DP Table: `c[n][amount]` → fill bottom-up
* Check which coins are used from the table

---

### 🎒 3. **0/1 Knapsack Problem**

* Items can't be broken (unlike fractional)
* Goal: Max value with weight limit
* Use DP table `V[i][w]`
* Trace back to see which items are included

---

### 🧭 4. **All-Pairs Shortest Path – Floyd's Algorithm**

* Find **shortest paths between all pairs** of nodes
* Update distance using each node as intermediate:

  ```
  D[i][j] = min(D[i][j], D[i][k] + D[k][j])
  ```

---

### 🔗 5. **Matrix Chain Multiplication**

* Goal: Find **best way to parenthesize** matrices to minimize multiplication
* Use DP table `M[i][j]` to store minimum cost
* Try all possible positions to split the chain

---

### 🔤 6. **Longest Common Subsequence (LCS)**

* Goal: Find max-length subsequence present in **both** strings
* If `A[i] == B[j]` →
  `LCS[i][j] = 1 + LCS[i-1][j-1]`
  Else →
  `LCS[i][j] = max(LCS[i-1][j], LCS[i][j-1])`

---

## 🎯 Memory Trick Table

| Problem                     | Goal                          | Key Idea                          |
| --------------------------- | ----------------------------- | --------------------------------- |
| Binomial Coefficient        | Count combinations            | Use recursive formula             |
| Make Change                 | Min coins for amount          | Bottom-up table fill              |
| 0/1 Knapsack                | Max value, weight ≤ limit     | Include or exclude each item      |
| Floyd’s Algorithm           | All-pairs shortest path       | Update via all intermediate nodes |
| Matrix Chain Multiplication | Min cost to multiply matrices | Try all parenthesizations         |
| LCS                         | Longest subsequence (common)  | Match → +1, Else → max(left, top) |

---

## 📚 DAA UNIT-5 – Simple Summary

### 🔗 **1. Graph Basics**

* **Graph = Vertices (nodes) + Edges (connections)**
* **Types**:

  * **Undirected** – Facebook friends
  * **Directed** – Web links (Page A → Page B)
* **Uses**: Google Maps, Facebook, Web crawling, OS resource graphs

---

### 🔍 **2. Graph Traversal**

| Traversal                      | Uses                    | Data Structure | Notes               |
| ------------------------------ | ----------------------- | -------------- | ------------------- |
| **DFS** (Depth First Search)   | Deep search (backtrack) | Stack          | May get stuck       |
| **BFS** (Breadth First Search) | Level-wise              | Queue          | Finds shortest path |

---

### 📜 **3. Topological Sort**

* **Used on DAGs (Directed Acyclic Graphs)**
* Sort tasks where one depends on another
* Example: Course Prerequisites

---

### 🔧 **4. Articulation Point**

* Removing such node **disconnects** the graph
* Important in **network design** (critical points)

---

### 🧠 **5. Backtracking**

* **Tries all possibilities**
* Goes back if stuck
* Example: **N-Queen Problem**

  * Place queens on board so no two attack
  * Time: O(N!)

---

### 🌳 **6. Branch and Bound**

* Solves **optimization problems** smartly
* Like backtracking but uses a **bound** to cut off bad paths
* Example: **0/1 Knapsack Problem**

---

### 🎮 **7. Mini-Max Principle**

* Used in **Game AI**
* MAX = human, tries to **maximize score**
* MIN = opponent, tries to **minimize your score**

---

### 🔤 **8. String Matching Algorithms**

| Algorithm                    | Best for          | Key Idea           |
| ---------------------------- | ----------------- | ------------------ |
| **Naive**                    | Small text        | Try all positions  |
| **Rabin-Karp**               | Big texts         | Hashing            |
| **Finite Automata**          | Fast matching     | Pattern as machine |
| **KMP** (Knuth-Morris-Pratt) | Repeated patterns | Prefix table (π)   |

---

### 🔒 **9. NP-Completeness**

| Class           | Meaning                                      |
| --------------- | -------------------------------------------- |
| **P**           | Solvable fast (polynomial time)              |
| **NP**          | Verifiable fast (solution is checkable)      |
| **NP-Complete** | Hardest in NP (e.g. **TSP**, **SAT**)        |
| **NP-Hard**     | Harder than NP, no known polynomial solution |

---

### 🚕 **10. Famous NP Problems**

* **Hamiltonian Cycle** – Visit all vertices once in a cycle
* **TSP** – Shortest path visiting all cities once and return

---

## 📝 Quick Revision Table

| Topic            | Example         | Key Point              |
| ---------------- | --------------- | ---------------------- |
| DFS / BFS        | Graph Search    | Stack vs Queue         |
| Topological Sort | Task Scheduling | DAG only               |
| N-Queens         | Backtracking    | Try all, undo if fail  |
| Knapsack         | Branch & Bound  | Cut bad paths early    |
| Mini-Max         | Tic Tac Toe     | MAX vs MIN turns       |
| String Matching  | DNA/Text search | Naive, KMP, RK, FA     |
| NP-Complete      | TSP, SAT        | No fast known solution |

---



