
% read the original data
originalData = readmatrix('originalData.txt');

% read the fitted Data
fittedData = readmatrix('fittedData.txt');

% read the state sequence
stateSequence = readmatrix('stateSequence.txt');

% the x-axis, is time step number, correspond to 0.05s 
% stepToTime = 0.05; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot figures
figure
t = tiledlayout(2,1); % 3 figures

% original data
time = originalData(1,:);
% x = x * stepToTime; 
ydata = originalData(2:4,:);
ax1 = nexttile;
a = plot(time, ydata(1,:),'-o','color',[0.7,0.91,0.41],'markersize',5,'linewidth',2);
a.Color(4) = 0.2;
hold on
b = plot(time, ydata(2,:),'-s','color',[0.07, 0.62, 1.0],'markersize',5,'linewidth',2);
b.Color(4) = 0.2;
c = plot(time, ydata(3,:),'-d','color',[1.0, 0.41, 0.16],'markersize',5,'linewidth',2);
c.Color(4) = 0.2;

xlim([0,ceil(max(time))]);
%ylabel('FRET');
%ylim([-0.2,1]);
%yticks(-0.2:0.2:1);
%set(gca,'linewidth', 2,'fontsize',20,'fontname','Times New Roman');

% fitted FRET 
% ax2 = nexttile;
x = fittedData(1,:);
yfit = fittedData(2:4,:);
plot(x,yfit(1,:),'color',[0.47,0.67,0.19], 'linewidth',2)
hold on
plot(x,yfit(2,:),'b', 'linewidth',2)
plot(x,yfit(3,:),'color',[0.64, 0.08, 0.18], 'linewidth',2)
xlim([0,ceil(max(x))]);
h1 = ylabel('FRET');
ylim([-0.2,1]);
yticks(-0.2:0.2:1);
set(gca,'linewidth', 2,'fontsize',20,'fontname','Times New Roman');

% state sequence
ax2 = nexttile;
x = stateSequence(1,:);
statesequence = stateSequence(2,:);
plot(x, statesequence,'-k','linewidth',2)
xlim([0,ceil(max(x))]);
h2 = ylabel('State');
ylim([-0.5,5.5]);
yticks(0:1:5);
set(gca,'linewidth', 2,'fontsize',20,'fontname','Times New Roman');

% manage the three figures: only keep the bottom x-axis 
linkaxes([ax1,ax2],'x');
%xlim([5,10]);
%xticks(5:1:10);
xlabel(t,'Time (s)','fontsize',20,'fontname','Times New Roman')
xticklabels(ax1,{})
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'unit','centimeters','position',[10 6 40 17.5]); 

% adjust the ylabel, to make their positions are in line
h1.Position
% h1.Position(1) = h2.Position(1);
h2.Position