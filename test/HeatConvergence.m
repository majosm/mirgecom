Globals1D;

N = 4;

t = 1.23456789;

errors = [];

for n = [4 8 16 32 64 128]

  [Nv, VX, K, EToV] = MeshGen1D(-pi/2,pi/2,n);

  StartUp1D;

  u = exp(-t)*cos(x);

  [rhs] = HeatCRHS1D(u,t);

  rhs_expected = -u;

  errors = [errors max(max(abs(rhs-rhs_expected)))];

  if length(errors) > 1
    order = log(errors(length(errors)-1)/errors(length(errors)))/log(2);
    fprintf("%24.16f %24.16f\n", errors(length(errors)), order);
  else
    fprintf("%24.16f\n", errors(length(errors)));
  end

end
