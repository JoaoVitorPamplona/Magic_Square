using MKL, LinearAlgebra, JuMP, SparseArrays, Cbc, BenchmarkTools, Dates, DataFrames, CSV, Plots, Polynomials

#Build Magic Square using optimization problem
function buildMS_OPT(n::Int64)
  ## Magic Constant
  σ = n * (n^2 + 1) / 2
  model = JuMP.Model(Cbc.Optimizer)
  ##Variables
  @variable(model, x[1:n, 1:n, 1:n^2], Bin)
  ## Each cell must contain a number
  @constraint(model, [i = 1:n, j = 1:n], sum(x[i, j, :]) == 1)
  ## Each number can be only in one cell
  @constraint(model, [k = 1:n^2], sum(x[:, :, k]) == 1)
  ## The sum of each row is equal to σ
  @constraint(model, [i = 1:n], sum(k * x[i, j, k] for j = 1:n for k = 1:n^2) == σ)
  ## The sum of each column is equal to σ
  @constraint(model, [j = 1:n], sum(k * x[i, j, k] for i = 1:n for k = 1:n^2) == σ)
  ## The sum of the diagonal  is equal to  σ
  @constraint(model, sum(k * x[i, i, k] for i = 1:n for k = 1:n^2) == σ)
  ## The sum of the second diagonal  is equal to σ
  @constraint(model, sum(k * x[n+1-i, i, k] for i = 1:n for k = 1:n^2) == σ)
  @objective(model, Min, 0)
  optimize!(model)
  xv = value.(x)
  ## Build Magic Square
  MS = zeros(n, n)
  for i = 1:n^2
    A = findfirst(x -> x == 1, xv[:, :, i])
    MS[A[1], A[2]] = i
  end
  MS = Int64.(MS)
  return MS
end

#Build Magic Square of order n that is double even 
function double_even_ms_old(n::Int64, Amin::Int64=1)
  A = reshape(collect(Amin:n^2+(Amin-1)), n, n)'
  n4 = div(n, 4)
  n2 = div(n, 2)
  M = [zeros(Int64, n4, n4) ones(Int64, n4, n2) zeros(Int64, n4, n4);
      ones(Int64, n2, n4) zeros(Int64, n2, n2) ones(Int64, n2, n4);
      zeros(Int64, n4, n4) ones(Int64, n4, n2) zeros(Int64, n4, n4)]
  multi = n^2 + 1
  return abs.(multi * M - A)
end

function double_even_ms(n::Int64, Amin::Int64=1)
  multi = n^2 + 1
  A_start = Amin
  result = Matrix{Int64}(undef, n, n)

  n4 = div(n, 4)
  n2 = div(n, 2)

  for i in 1:n, j in 1:n
      a_val = A_start + (i - 1) * n + (j - 1)
      m_val = (
          (i ≤ n4 || i > n - n4) && (j > n4 && j ≤ n - n4) ||
          (i > n4 && i ≤ n - n4) && (j ≤ n4 || j > n - n4)
      ) ? 1 : 0
      result[i, j] = abs(multi * m_val - a_val)
  end

  return result
end


#Auxiliar function for double_even_ms() function
function swap_blocks!(A, nRows, nCols, RowX, ColX, RowY, ColY)
  if nCols == 0
    return
  end
  for J = 0:nCols-1
    @inbounds for I = 0:nRows-1
      A[RowX+I, ColX+J], A[RowY+I, ColY+J] = A[RowY+I, ColY+J], A[RowX+I, ColX+J]
    end
  end
end

#Build Magic Square of order n that is odd
function odd_ms(n::Int64, Amin::Int64=1)
  A = zeros(Int64, n, n)
  i = 1
  j = div((n - 1), 2) + 1
  e = Amin
  u = n^2 + (Amin - 1)
  A[i, j] = e
  while e < u
    e = e + 1
    ui = i
    uj = j
    i -= 1
    j += 1
    if i == 0
      i = n
    end
    if j > n
      j = 1
    end
    if A[i, j] != 0
      i = ui + 1
      j = uj
    end
    A[i, j] = e
  end
  return A
end

#Build Magic Square of order n that is single even
function single_even_ms(n::Int64, Amin::Int64=1)
  block_order_x = div(n, 2)
  block_element_x = div(n^2, 4)
  A = [odd_ms(block_order_x, Amin) odd_ms(block_order_x, 2 * block_element_x + Amin);
    odd_ms(block_order_x, 3 * block_element_x + Amin) odd_ms(block_order_x, block_element_x + Amin)]

  swap_blocks!(A, div(n - 2, 4), div(n - 2, 4), 1, 1, div(n + 2, 2), 1)
  swap_blocks!(A, div(n - 2, 4), div(n - 2, 4), div(n + 6, 4), 1, div(3n + 6, 4), 1)
  swap_blocks!(A, 1, div(n - 2, 4), div(n + 2, 4), div(n + 2, 4), div(3n + 2, 4), div(n + 2, 4))
  swap_blocks!(A, div(n, 2), div(n - 6, 4), 1, div(3n + 10, 4), div(n + 2, 2), div(3n + 10, 4))
  return A
end

#Compute de magic constant of a magic square
function constMagic(n::Int64, Amin::Int64=1)
  Amax = Amin + (n^2) - 1
  return div(n * (n^2 + 1), 2) + n * (Amax - (n^2))
end

#Verify if a matrix is a magic square
function ismagic(A::Matrix)
  if typeof(A) != Matrix{Int64}
    error("Matrix coefficients are not integers. Matrix is not a magic square.")
  end

  m, n = size(A)
  m != n ? error("Matrix order $m by $n is not square. Matrix is not a magic square.") :
  Amin = minimum(A)
  Amax = maximum(A)
  MagicConst = constMagic(n, Amin)
  sumcols = sum(A, dims=1)
  sumrows = sum(A, dims=2)
  sumdiag = tr(A)
  sumdiag != MagicConst ? error("Matrix of order $n is not a magic square.(main diagonal)") :
  sumdiagsec = sum(A[n:n-1:end-1])
  sumdiagsec != MagicConst ? error("Matrix of order $n is not a magic square.(secondary diagonal)") :
  for i in 1:n
    if sumrows[i] != MagicConst || sumcols[i] != MagicConst
      error("Matrix of order $n is not a magic square.(row or col $i)")
    end
  end
  print("This matrix is a magic square!\n")
end

#Build Magic Square of order n
function ms(n::Int64, Amin::Int64=1)
  if isodd(n)
    A = odd_ms(n, Amin)
  elseif n % 4 == 0
    A = double_even_ms(n, Amin)
  else
    A = single_even_ms(n, Amin)
  end
end




#Examples of use
buildMS_OPT(3)
constMagic(3)
ms(3)
ismagic(ms(3))



