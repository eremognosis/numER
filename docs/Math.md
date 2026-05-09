# How it works (My Fav) 
(Author: *eremognosis*, ofc)
## Why I chose things the way I did?
#### If you were sleeping, what I did?
- $n$ = `ndim` (the number of axes)
- Shape $$S = (S_0,S_1,\dots, S_{n-1})$$ where each $S_k$ are natural numbers. The Shape tuple for $n$ dimensional array is just a way of saying in each dimension how long it would be. Sounded ambiguous? Let's take this cute matrix (a $2$ dimensional array) $$\begin{pmatrix}

a & b & c \\

d & e & f

\end{pmatrix} 
$$ which on memory which is famously linear is just 6 addresses linearly like ${a,b,c,d,e,f}$ . So what is dimension here? the number of rows and columns!! Its shape will be ${2,3}$. How it makes sense? It does because our C-row-major shape (where the number of rows come first) so in thsi the $S[0]$ and $S[1]$ are dimensions (total dimensions = 2 = sum of shape elements)
- Strides $$r=(r_0,r_1,\dots,r_{n-1})$$ where $r_k$ are whole numbers. Now what are they? As I said (it's not me who said it for the first time though) that the computer memory is linear and pointer to pointer is one of the certified method of performance low down. So we store in linear manner. We are flat-earther when it comes to memory!! So for an $n$ dimensional array every element has an unique address $(i_0,i_1,...,i_{n-1})$ which we don't care. We need its address in memory. So strides is our way of looking at the linear data in memory. Stride of a particular dimension is the number of spots (in memory blocks of each element) we need to move to move by 1 unit in particular dimension. In our matrix example, the memory representation is $$ a \  b \ c \ d \ e \ f  \\ $$ at addresses 0 to 5. Lets say we want to move one unit in `S[0]` i.e. across rows (along columns) example $b$ to $e$ , how many elements we have to move  in the linear memory? 3. Now lets say accross columns lets say $e$ to $f$ how many spaces? 2. Hence the stride tuple for this matrix (2d array) is $(3,2)$ 


## Contiguous Stride Formula (Your College didn't teach)
### Statement: 
For a row-major layout:
$$ 
\begin{align}

r_{n-1}=d \\ 
r_k = r_{k+1}*S_{k+1}, \\ \ \text{for }k=n-2,n-3,\dots,0

\end{align}
$$
#### Proof:
Let, our size of each element in memory (block) is $d$
Consider the flat index $m$  mapping to multi-index $i_k$. The memory offset from $B'$ is $$\sum i_K \times r_k$$; to ensure consecutive flat indices are $d$ bytes apart when varying the last index only, set $r_{n-1}=d$; 
to ensure proper spacing when carrying into higher axes, choose $r_k$ proportionally as above.
In Words : To get to a particular coordinate, sum over all axes times its strides (total memory blocks)!

How we get back? 
```pseudo
For k from n-1 down to 0:
	i_k = x mod S_k
	x = floor(x / S_k)
```
This decomposition is consistent with the contiguous stride formula: the byte address difference between $x$ and $x+1$ equals `d` if and only if $r_{n-1}=d$ and the other $r_{k}$ follow the product rule.

### **General (non-contiguous) layouts and views**

An `NDArray` may have arbitrary `strides` (including zeros). The address of an index tuple is still

$$A(i) = B' + \sum i_k * r_k$$

but contiguity is the special case where $r$ obeys the contiguous formula. Views (slices, reshapes without copy) share the same `data` base but may have different `shape`, `strides`, and `offset` while still satisfying the address formula above.

  
### The Alignment Isomorphism

We start with $N$ arrays where each $S^{(a)}$ has its own dimensionality. To perform any element-wise operation, we must map them to a Common Broadcasted Domain $\mathbb{D}$. We force all shapes to the same rank $m$ by prepending 1s:

$$S^{(a)}_{aligned} = (\underbrace{1, 1, \dots, 1}_{m - n_a \text{ times}}, S^{(a)}_0, \dots, S^{(a)}_{n_a-1})$$

The resulting shape $O$ is the least upper bound of all aligned shapes in the lattice of shapes under the broadcasting partial order.

  
### Fake-Stride Construction (The Null-Space Mapping)

The stride $R^{(a)}_k$ is a piecewise function that determines how the pointer $P^{(a)}$ moves through the physical buffer as we iterate through the logical domain $O$.

$$R^{(a)}_k =

\begin{cases}

r^{(a)}_{k'} & \text{if } S^{(a)}_{aligned}[k] = O_k \\

0 & \text{if } S^{(a)}_{aligned}[k] = 1

\end{cases}$$

When $R^{(a)}_k = 0$, the partial derivative of the address function with respect to the $k$-th index is zero: $\frac{\partial P^{(a)}}{\partial i_k} = 0$. This mathematically collapses that dimension, meaning the entire dimension $O_k$ is projected onto a single memory location.

  
### Iterator Pointer Algebra & Lexicographic Rolling

To avoid the computational cost of the full summation $\sum i_k \times R_k$ inside a hot loop, we use differential pointer updates.

For a lexicographic iterator (right-to-left, right is the smallest step), we only increment the pointer by $R_{m-1}$ at each step. However, when an axis $k$ "rolls over" (reaches $O_k - 1$), we must reset it. The Backstride is the corrective displacement:

$$\text{back}^{(a)}_k = R^{(a)}_k \times (O_k - 1)$$

The pointer update is then $P^{(a)} = P^{(a)} - \text{back}^{(a)}_k$. This is essentially a Discrete Integral over the dimension being reset.

  
### The Generalized Address Formula

The formula $A(i) = B' + \sum i_k \times r_k$ is the Grand Unified Theory of `NDArray`.

$B'$: The Effective Base. This is the pointer to the first element of thhe view, which might actually be in the middle of the original memory block ($B + \text{offset}$). We literally can start from anywhere. the whole idea was to instead of moving data (costly) we change the way how we look at it (suddenly 6 dim vector can become 2x3 matrix).

$r_k$: The Arbitrary Stride. In a "normal" array, these are fixed by the shape. In a View, these can be anything.

If $r_k = 0$, we are Broadcasting. Remember what I told about strides? Its number of blocks in each dimension to get the specified coordinate. What if we make it 0? That means  dont move along this direction, keep the same! Which is broadcasting.