DGEMV
in
  A : matrix(column), x : vector(column), alpha : scalar, beta : scalar
inout
  y : vector(column)
{
  y = alpha*(A'*x) + beta*y
}
