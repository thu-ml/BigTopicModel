clear all;
close all;
clc;
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');



% ------------------------ NYSmaller ----------------------------
files = {...
'quality_nysmaller/result_threshold_10000000_n_mc_iters_-1.log',...
'quality_nysmaller/result_threshold_10000000_n_mc_iters_30.log',...
'quality_nysmaller/result_threshold_50_n_mc_iters_-1.log',...
'quality_nysmaller/result_threshold_50_n_mc_iters_30.log',...
'quality_nysmaller/result_threshold_-1_n_mc_iters_-1.log',...
'quality_nysmaller/result_threshold_-1_n_mc_iters_30.log',...
'quality_nysmaller/hlda_c.log'};

color = fliplr(repmat({[0.5 0 0], [1 0 0], [0 0.5 0], [0 1 0], [0 0 0.5], [0 0 1], [0 0 0]}, 1, 3));
draw_box(5, files, [33, 66, 100], 5, 6, ...
{'(0, 33]', '(33, 66]', '(66, 100]'}, ...
color, {'CGS', 'CGS$^a$', 'PCGS', 'PCGS$^a$', 'BGS', 'BGS$^a$', 'hlda-c'}, 'quality', 'vertical', 'eastoutside', [2000 3500]);

% ------------------------ NIPS ----------------------------
files = {...
'quality_nips/result_threshold_10000000_n_mc_iters_-1.log',...
'quality_nips/result_threshold_10000000_n_mc_iters_30.log',...
'quality_nips/result_threshold_50_n_mc_iters_-1.log',...
'quality_nips/result_threshold_50_n_mc_iters_30.log',...
'quality_nips/result_threshold_-1_n_mc_iters_-1.log',...
'quality_nips/result_threshold_-1_n_mc_iters_30.log',...
'quality_nips/hlda_c.log'};

color = fliplr(repmat({[0.5 0 0], [1 0 0], [0 0.5 0], [0 1 0], [0 0 0.5], [0 0 1], [0 0 0]}, 1, 3));
draw_box(5, files, [33, 66, 100], 5, 6, ...
{'(0, 33]', '(33, 66]', '(66, 100]'}, ...
color, {'CGS', 'CGS$^a$', 'PCGS', 'PCGS$^a$', 'BGS', 'BGS$^a$', 'hlda-c'}, 'quality-nips', 'vertical', 'eastoutside', [1900 2800]);

% ----------------------- Iters ----------------------------

files = {'../../BTM-2/work/result_n_mc_iters_1.log','../../BTM-2/work/result_n_mc_iters_2.log','../../BTM-2/work/result_n_mc_iters_4.log','../../BTM-2/work/result_n_mc_iters_8.log','../../BTM-2/work/result_n_mc_iters_16.log','../../BTM-2/work/result_n_mc_iters_32.log','../../BTM-2/work/result_n_mc_iters_64.log'};
color = cell(7, 1);
for i = 1:7
    color{i} = hsv2rgb([2.0 / 3 * (i-1) / 6  1  1]);
end
color = fliplr(repmat(color, 1, 3));
draw_box(6, files, [50, 100, 150], 6, 7, ...
{'(0, 50]', '(50, 100]', '(100, 150]'}, ...
color, {'I=1', 'I=2', 'I=4', 'I=8', 'I=16', 'I=32', 'I=64'}, 'iters', 'vertical', 'northeast', [1900 2500]);

% ----------------------- M --------------------------
m = dlmread('m.log');
for i = 1:size(m, 1)
    if m(i, 1) == 1000000
        m(i, 1) = 131072;
    end
end

%subplot(2, 2, 1);
fig = figure(21);
spacing = 0.15;

boxplot(m(:, 3), log2(m(:, 1)));
xlabel('M');
ylabel('perplexity');
set(gca, 'XTick', [1, 6, 11, 18]);
%xlim([0, 17]);
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'XTickLabel', {'2^0', '2^5', '2^{10}', '\infty'});
set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'm_1', 'pdf');

%pos = get(gca, 'Position');
%pos(1) = 0.055;
%pos(3) = 0.9;
%set(gca, 'Position', pos);

fig = figure(22);
group_cnt = zeros(18, 1);
group_i = zeros(18, 1);
group_c = zeros(18, 1);
for i = 1 : size(m, 1)
    g = log2(m(i, 1)) + 1;
    group_cnt(g) = group_cnt(g) + 1;
    group_i(g) = group_i(g) + m(i, 11);
    group_c(g) = group_c(g) + m(i, 12);
end
group_i = group_i ./ group_cnt;
group_c = group_c ./ group_cnt;
bar([group_i group_c], 'stacked');
xlabel('M');
ylabel('topics');
ylim([0, 1000]);
xlim([0, 19]);
legend('I', 'C');
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'XTick', [1 6 11 18]);
set(gca, 'XTickLabel', {'2^0', '2^5', '2^{10}', '\infty'});
set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'm_2', 'pdf');
%
fig = figure(23);
group_cnt = zeros(18, 1);
group_times = zeros(18, 3);
for i = 1 : size(m, 1)
    g = log2(m(i, 1)) + 1;
    group_cnt(g) = group_cnt(g) + 1;
    group_times(g, :) = group_times(g, :) + m(i, 7:9);
end
group_times = group_times ./ repmat(group_cnt, 1, 3);
bar(group_times, 'stacked');
xlim([0, 19]);
xlabel('M');
ylabel('CPU time (s)');
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'XTick', [1 6 11 18]);
set(gca, 'XTickLabel', {'2^0', '2^5', '2^{10}', '\infty'});
legend('I', 'C', 'Z', 'Location', 'northwest');
set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'm_3', 'pdf');
%
fig = figure(24);
group_cnt = zeros(18, 1);
group_comm = zeros(18, 1);
for i = 1 : size(m, 1)
    g = log2(m(i, 1)) + 1;
    group_cnt(g) = group_cnt(g) + 1;
    group_comm(g) = group_comm(g) + m(i, 6);
end
group_comm = group_comm ./ group_cnt;
plot(1:18, group_comm);
xlabel('M');
ylabel('sync.s / second');
xlim([0, 19]);
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'XTick', [1 6 11 18]);
set(gca, 'XTickLabel', {'2^0', '2^5', '2^{10}', '\infty'});
set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'm_4', 'pdf');

% TODO num. collapsed vs num. instantiated
% TODO amt. of communication

% -------------------- Num docs ------------------
num_docs = dlmread('num_docs.log');
num_docs = wrev(num_docs);

fig = figure(31);

loglog(num_docs);
xlim([0, length(num_docs)]);
xlabel('rank');
ylabel('number of documents');
set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'num_docs_1', 'pdf');

fig = figure(32);
plot(cumsum(num_docs));
hold on;
plot([426, 426], [0, 10000000], 'r--');         % threshold = 128: 99.45% documents
xlim([0, length(num_docs)]);
ylim([0, sum(num_docs)]);
xlabel('rank');
ylabel('cumulative number of documents');

set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'num_docs_2', 'pdf');

% ------------------ Time -----------------------
time = dlmread('time.log');

fig = figure(4);
bar(time, 'FaceColor', [0.5 0.5 1]);
set(gca, 'yscale', 'log');
set(gca, 'XTickLabel', {'hlda-c', 'CGS$^a$', 'PCGS$^a$', 'BGS$^a$'});
ylabel('Time (seconds)');

set(gcf, 'PaperPosition', [0 0 5 2]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 2]); %Keep the same paper size
saveas(gcf, 'time', 'pdf');

% ----------------------- Threads ----------------------------

fig = figure(71);
hold off;
thr = dlmread('threads/nytimes.log');
boxplot(thr(:, 2), thr(:, 1));
xlabel('Number of threads');
ylabel('Perplexity');
ylim([3700 4100]);
set(gca, 'FontSize', 8);
set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'threads_1', 'pdf');

fig = figure(72);
serial_time = mean(thr(thr(:, 1) == 1, 3));
boxplot(serial_time ./ thr(:, 3), thr(:, 1));
hold on;
xlabel('Number of threads');
ylabel('SpeedUp');
set(gca, 'FontSize', 8);

plot(1:12, 1:12, 'r--');
c = get(gca, 'Children');
hleg1 = legend(c(1), 'Ideal', 'Location', 'northwest');
ylim([0, 12]);

set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'threads_2', 'pdf');

% ----------------------- Machines-small ----------------------------
fig = figure(81);
hold off;
thr = dlmread('machines/pubmed.log');
boxplot(thr(:, 2), thr(:, 1));
xlabel('Number of machines');
ylabel('Perplexity');
ylim([2950 3350]);
set(gca, 'FontSize', 8);
set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'machines-small-1', 'pdf');

fig = figure(82);
serial_time = mean(thr(thr(:, 1) == 1, 3));
boxplot(serial_time ./ thr(:, 3), thr(:, 1));
hold on;
xlabel('Number of machines');
ylabel('SpeedUp');
set(gca, 'FontSize', 8);

plot(1:10, 1:10, 'r--');
c = get(gca, 'Children');
hleg1 = legend(c(1), 'Ideal', 'Location', 'northwest');
ylim([0, 10]);

set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'machines-small-2', 'pdf');

% ----------------------- Machines-large ----------------------------
fig = figure(101);
hold off;
thr = dlmread('machines-large.log');
boxplot(thr(:, 2), thr(:, 1));
xlabel('Number of machines');
ylabel('Perplexity');
ylim([2900 3700]);
set(gca, 'FontSize', 8);
set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'machines-large-1', 'pdf');

fig = figure(102);
serial_time = mean(thr(thr(:, 1) == 10, 3));
boxplot(serial_time ./ thr(:, 3), thr(:, 1));
hold on;
xlabel('Number of machines');
ylabel('SpeedUp');
set(gca, 'FontSize', 8);

plot(1:10, 1:10, 'r--');
c = get(gca, 'Children');
hleg1 = legend(c(1), 'Ideal', 'Location', 'northwest');
ylim([0, 10]);

set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'machines-large-2', 'pdf');

% ---------------------- MC Samples ----------------------------

files = {'n_mc_samples/result_n_mc_samples_1.log',...
'n_mc_samples/result_n_mc_samples_2.log',...
'n_mc_samples/result_n_mc_samples_4.log',...
'n_mc_samples/result_n_mc_samples_8.log',...
'n_mc_samples/result_n_mc_samples_16.log',...
'n_mc_samples/result_n_mc_samples_32.log',...
'n_mc_samples/result_n_mc_samples_64.log',...
'n_mc_samples/result_n_mc_samples_128.log'};

num_topics = [];
groups = [];
perplexity = [];

num_files = numel(files);
for i = 1:num_files
    dat = dlmread(files{i});
    num_topics = [num_topics; dat(:, 7)];
    perplexity = [perplexity; dat(:, 8)];
    groups = [groups; repmat([2^(i-1)], 10, 1)];
end

fig = figure(91);
hold off;
boxplot(num_topics, groups);
xlabel('$S$');
ylabel('Number of topics');

set(gca, 'FontSize', 8);
set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'n-mc-samples-1', 'pdf');

fig = figure(92);
hold off;
boxplot(perplexity, groups);
xlabel('$S$');
ylabel('Perplexity');
set(gca, 'FontSize', 8);
set(gcf, 'PaperPosition', [0 0 2.5 2.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [2.5 2.5]); %Keep the same paper size
saveas(gcf, 'n-mc-samples-2', 'pdf');
