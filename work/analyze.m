clear all;

cs = dlmread('result_threshold_10000000_n_mc_iters_-1.log');
csa = dlmread('result_threshold_10000000_n_mc_iters_30.log');
pcs = dlmread('result_threshold_50_n_mc_iters_-1.log');
pcsa = dlmread('result_threshold_50_n_mc_iters_30.log');
bgs = dlmread('result_threshold_-1_n_mc_iters_-1.log');
bgsa = dlmread('result_threshold_-1_n_mc_iters_30.log');
hldac = dlmread('hlda_c.log');

hold off;
fig = figure(1);
plot(cs(:,5), cs(:,6), 'ro');
hold on;
plot(csa(:,5), csa(:,6), 'rx');

plot(pcs(:,5), pcs(:,6), 'go');
plot(pcsa(:, 5), pcsa(:,6), 'gx');

plot(bgs(:,5), bgs(:,6), 'bo');
plot(bgsa(:, 5), bgsa(:,6), 'bx');
%plot(pcsmctrue(:, 1), pcsmctrue(:,2), 'gv');

%plot(ismctrue(:, 1), ismctrue(:,2), 'bv');
%
plot(hldac(:,3), hldac(:,4), 'c^');

xlim([0 100]);
ylim([2000, 3200]);

xlabel('number of topics');
ylabel('perplexity');

%legend('cs', 'cs (mc)', 'pcs', 'pcs (mc)', 'pcs (mc, dir)', 'is', 'is (mc)', 'ic (mc, dir)', 'hlda-c');
legend('CGS', 'CGS^a', 'PCGS', 'PCGS^a', 'BGS', 'BGS^a', 'hlda-c');

set(gcf, 'PaperPosition', [0 0 5 5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 5]); %Keep the same paper size
saveas(gcf, 'quality', 'pdf');
