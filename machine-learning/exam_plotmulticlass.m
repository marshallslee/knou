X = randn(90, 2);	% sample data
Y = [ones(30,1); ones(30,1)*2; ones(30,1)*3];	% class label {1, 2, 3} three class

figure, hold on, grid on
for ii = 1 : 3
	if exist("OCTAVE_VERSION", "builtin")>0 % Octave
		plot(X(Y==ii,1), X(Y==ii,2), '.', 'MarkerSize', 20)
	else % MATLAB
		plot(X(Y==ii,1), X(Y==ii,2), '.')
	end
end
legend()