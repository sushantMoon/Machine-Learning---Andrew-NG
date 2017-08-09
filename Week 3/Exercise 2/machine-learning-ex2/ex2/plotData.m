function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Lorenzo Feliz

% Finding the indices for accepted applicants (y=11) and rejected applicants (y=0)
ac = find(y == 1);% accepted students
re = find(y == 0);% rejected students

% Plotting the accepted students
plot( X(ac,1), X(ac,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
% Plotting the rejected students
plot( X(re,1), X(re,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);






% =========================================================================



hold off;

end
