Optimization problem with equality constraints:

$$ \underset{\mathbf{a}, \mathbf{b}, \mathbf{c}}{\mathrm{minimize}} \quad f = \sum_{i=0}^{n-1} \int_0^{{\Delta k}_i} (p_i(t) - y_i)^2 \, \mathrm{d}{t} $$

$$ g_{1i} = \frac{1}{{\Delta k}_i} \int_0^{{\Delta k}_i} p_i(t) \, \mathrm{d}t - y_i = 0 \quad \text{for } i \text{ in } \{0, \ldots, n-1\} $$

$$ g_{2i} = p_{i+1}(0) - p_i({\Delta k}_i) = 0 \quad \text{for } i \text{ in } \{0, \ldots, n-2\} $$

$$ g_{3i} = p_{i+1}'(0) - p_i'({\Delta k}_i) = 0 \quad \text{for } i \text{ in } \{0, \ldots, n-2\} $$

Where:

$$ {\Delta k}_i = k_{i+1} - k_i $$

$$ p_i(t) = a_i t^2 + b_i t + c_i $$

$$
\begin{aligned}
(p_i(t) - y_i)^2
& = p_i(t)^2 - 2 p_i(t) y_i + y_i^2 \\
& = (a_i t^2 + b_i t + c_i)^2 - 2 (a_i t^2 + b_i t + c_i) y_i + y_i^2 \\
& = a_i^2 t^4 + b_i^2 t^2 + 2 a_i t^3 b_i + 2 a_i t^2 c_i + 2 b_i c_i t + c_i^2 - 2 a_i t^2 y_i - 2 b_i t y_i - 2 c_i y_i + y_i^2 \\
& = (a_i^2) t^4 + (2 a_i b_i) t^3 + (b_i^2 + 2 a_i c_i - 2 a_i y_i) t^2 + (2 b_i c_i - 2 b_i y_i) t + (c_i^2 - 2 c_i y_i + y_i^2)
\end{aligned}
$$

$$
\begin{aligned}
\int_0^{{\Delta k}_i} (p_i(t) - y_i)^2 \, \mathrm{d}{t}
& = \left[ \frac{a_i^2}{5} t^5 + \frac{a_i b_i}{2} t^4 + \frac{b_i^2 + 2 a_i (c_i - y_i)}{3} t^3 + b_i (c_i - y_i) t^2 + (c_i^2 - 2 c_i y_i + y_i^2) t + C \right]_{0}^{{\Delta k}_i} \\
& = \frac{a_i^2}{5} ({\Delta k}_i)^5 + \frac{a_i b_i}{2} ({\Delta k}_i)^4 + \frac{b_i^2 + 2 a_i (c_i - y_i)}{3} ({\Delta k}_i)^3 + b_i (c_i - y_i) ({\Delta k}_i)^2 + (c_i^2 - 2 c_i y_i + y_i^2) {\Delta k}_i
\end{aligned}
$$

$$ p_i'(t) = 2 a_i t + b_i $$

$$
\begin{aligned}
g_{1i}
& = \frac{1}{{\Delta k}_i} \int_0^{{\Delta k}_i} p_i(t) \, \mathrm{d}t - y_i \\
& = \frac{1}{{\Delta k}_i} \left[ \frac{a_i}{3} t^3 + \frac{b_i}{2} t^2 + c_i t + C \right]_0^{{\Delta k}_i} - y_i \\
& = \frac{a_i}{3} ({\Delta k}_i)^2 + \frac{b_i}{2} {\Delta k}_i + c_i - y_i \\
g_{2i} & = p_{i+1}(0) - p_i({\Delta k}_i) = c_{i+1} - (a_i ({\Delta k}_i)^2 + b_i {\Delta k}_i + c_i) \\
g_{3i} & = p_{i+1}'(0) - p_i'({\Delta k}_i) = b_{i+1} - (2 a_i {\Delta k}_i + b_i)
\end{aligned}
$$

Converted to unconstrained optimization problem using lagrange multipliers:

$$ \underset{\mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{l}_1, \mathbf{l}_2, \mathbf{l}_3}{\mathrm{minimize}} \quad \mathcal{L} = f + \mathbf{l}_1 \cdot \mathbf{g}_1 + \mathbf{l}_2 \cdot \mathbf{g}_2 + \mathbf{l}_3 \cdot \mathbf{g}_3 $$

For $i$ in $\{0, \ldots, n-1\}$:

$$
\begin{aligned}
& \frac{\partial \mathcal{L}}{\partial a_i} = 0 \\
& \implies \frac{\partial f}{\partial a_i} + l_{1i} \frac{\partial g_{1i}}{\partial a_i} + l_{2i} \frac{\partial g_{2i}}{\partial a_i} + l_{3i} \frac{\partial g_{3i}}{\partial a_i} = 0 \\
& \implies \left( \frac{2}{5} a_i ({\Delta k}_i)^5 + \frac{1}{2} b_i ({\Delta k}_i)^4 + \frac{2}{3} (c_i - y_i) ({\Delta k}_i)^3 \right) \\
& \phantom{\implies} {} + l_{1i} \left( \frac{1}{3} ({\Delta k}_i)^2 \right) + l_{2i} \left( -({\Delta k}_i)^2 \right) + l_{3i} \left( -2 {\Delta k}_i \right) \\
& \phantom{\implies} {} = 0 \\
& \implies \left( \frac{2}{5} ({\Delta k}_i)^5 \right) a_i + \left( \frac{1}{2} ({\Delta k}_i)^4 \right) b_i + \left( \frac{2}{3} ({\Delta k}_i)^3 \right) c_i \\
& \phantom{\implies} {} + \left( \frac{1}{3} ({\Delta k}_i)^2 \right) l_{1i} + \left( -({\Delta k}_i)^2 \right) l_{2i} + \left( -2 {\Delta k}_i \right) l_{3i} \\
& \phantom{\implies} {} = \frac{2}{3} y_i ({\Delta k}_i)^3
\end{aligned}
$$

$$A_4 \mathbf{x} = \mathbf{b}_4$$

For $i$ in $\{0, \ldots, n-1\}$:

$$
\begin{aligned}
& \frac{\partial \mathcal{L}}{\partial b_i} = 0 \\
& \implies \frac{\partial f}{\partial b_i} + l_{1i} \frac{\partial g_{1i}}{\partial b_i} + l_{2i} \frac{\partial g_{2i}}{\partial b_i} + l_{3i} \frac{\partial g_{3i}}{\partial b_i} + l_{3,i-1}^* \frac{\partial g_{3,i-1}}{\partial b_i} = 0 \\
& \implies \left( \frac{1}{2} a_i ({\Delta k}_i)^4 + \frac{2}{3} b_i ({\Delta k}_i)^3 + (c_i - y_i) ({\Delta k}_i)^2 \right) \\
& \phantom{\implies} {} + l_{1i} \left( \frac{1}{2} {\Delta k}_i \right) + l_{2i} \left( -{\Delta k}_i \right) + l_{3i} \left( -1 \right) + l_{3,i-1}^* \left( 1 \right) \\
& \phantom{\implies} {} = 0 \\
& \implies \left( \frac{1}{2} ({\Delta k}_i)^4 \right) a_i + \left( \frac{2}{3} ({\Delta k}_i)^3 \right) b_i + \left( ({\Delta k}_i)^2 \right) c_i \\
& \phantom{\implies} {} + \left( \frac{1}{2} {\Delta k}_i \right) l_{1i} + \left( -{\Delta k}_i \right) l_{2i} + \left( -1 \right) l_{3i} + \left( 1 \right) l_{3,i-1}^* \\
& \phantom{\implies} {} = y_i ({\Delta k}_i)^2
\end{aligned}
$$

\* Note: $l_{3,i-1}$ is considered zero when $i=0$.

$$A_5 \mathbf{x} = \mathbf{b}_5$$

For $i$ in $\{0, \ldots, n-1\}$:

$$
\begin{aligned}
& \frac{\partial \mathcal{L}}{\partial c_i} = 0 \\
& \implies \frac{\partial f}{\partial c_i} + l_{1i} \frac{\partial g_{1i}}{\partial c_i} + l_{2i} \frac{\partial g_{2i}}{\partial c_i} + l_{2,i-1} \frac{\partial g_{2,i-1}}{\partial c_i} + l_{3i} \frac{\partial g_{3i}}{\partial c_i} = 0 \\
& \implies \left( \frac{2}{3} a_i ({\Delta k}_i)^3 + b_i ({\Delta k}_i)^2 + 2 (c_i - y_i) {\Delta k}_i \right) \\
& \phantom{\implies} {} + l_{1i} \left( 1 \right) + l_{2i} \left( -1 \right) + l_{2,i-1} \left( 1 \right) + l_{3i} \left( 0 \right) \\
& \phantom{\implies} {} = 0 \\
& \phantom{\implies} \left( \frac{2}{3} ({\Delta k}_i)^3 \right) a_i + \left( ({\Delta k}_i)^2 \right) b_i + \left( 2 {\Delta k}_i \right) c_i \\
& \phantom{\implies} {} + \left( 1 \right) l_{1i} + \left( -1 \right) l_{2i} + \left( 1 \right) l_{2,i-1} \\
& \phantom{\implies} {} = 2 y_i {\Delta k}_i
\end{aligned}
$$

\* Note: $l_{2,i-1}$ is considered zero when $i=0$.

$$A_6 \mathbf{x} = \mathbf{b}_6$$
