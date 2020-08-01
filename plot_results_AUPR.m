clc;
clear all;
close all;
set(0,'defaultLineLineWidth',1.5)
set(0,'defaulttextInterpreter','latex') 
set(0,'defaultAxesFontSize',10) 
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

tbl=readtable('output/D4_100_1_timeseries_3D_save1_GenePairs.csv');
tbl1=readtable('output/D4_100_5_timeseries_3D_save1_GenePairs.csv');
h_r=tbl1{:,47:49};
h_t=tbl1{:,50:52};
for i=1:3520
    dh(i)=norm(h_r(i,:)-h_t(i,:));
end

label=tbl.Var46;
regexp=zscore(tbl{:,4:24}');
tgexp=zscore(tbl{:,25:45}');

histogram(dh(find(label==1)));
hold on;
alpha(0.5)
histogram(dh(find(label==0)));
xlabel('Latent space represenatation of input data')
l=legend('Regulatory connection','No regulatory connection');
l.Box='off'

RHO = corr(regexp,tgexp);
rho=diag(RHO);

figure;
subplot(2,2,1);
histogram(rho(find(label==1)));
hold on;
alpha(0.5);
histogram(rho(find(label==0)));
hold on;
%legend('Regulatory connection','No regulatory connection')
xlabel('Correlation')



RHO = corr(regexp(1:20,:),tgexp(2:21,:));
rho=diag(RHO);
subplot(2,2,2);
histogram(rho(find(label==1)));
hold on;
alpha(0.5);
histogram(rho(find(label==0)));
hold on;
%legend('Regulatory connection','No regulatory connection')
xlabel('Time lagged correlation')

for i=1:length(rho)
minf(i)=FastPairMI_pseudo_code_version([regexp(:,i)',tgexp(:,i)'],0.3);
end

%figure;
subplot(2,2,3);
histogram(minf(find(label==1)));
hold on;
alpha(0.5);
histogram(minf(find(label==0)));
hold on;
legend('Regulatory connection','No regulatory connection')
xlabel('Mutual Information')
l=legend('Regulatory connection','No regulatory connection');
l.Box='off';
newPosition = [0.4 0.2 0.7 0.2];
newUnits = 'normalized';

set(l,'Position', newPosition,'Units', newUnits);
SaveImagePdf(gcf,4.1,4.1,'corr_MI.pdf','pdf','in')

figure;
plot(1:21,regexp(:,1)); hold on;
plot(1:21,tgexp(:,1))


figure;
plot(1:21,regexp(:,3000)); hold on;
plot(1:21,tgexp(:,3000))


for j=1:5
fnamejs=['output/stats_D4_100_',num2str(j),'_t.json'];
val=jsondecode(fileread(fnamejs));

score=val.pred;
target=val.true;
[prec, tpr, fpr, thresh]=prec_rec(score, target);
figure;
%plot(prec,tpr); hold on;
aupr_cnn(j)=abs(trapz(tpr,prec));
auroc_cnn(j)=abs(trapz(fpr,tpr));
end


for j=1:5
fnamejs=['output/RNN_stats_n',num2str(j),'_32_60_adam_dropout.json'];
val=jsondecode(fileread(fnamejs));

score=val.pred;
target=val.true;
[prec, tpr, fpr, thresh]=prec_rec(score, target);
%figure;
%plot(prec,tpr); hold on;
aupr_rnn(j)=abs(trapz(tpr,prec));
auroc_rnn(j)=abs(trapz(fpr,tpr));
figure(j)
plot(fpr,tpr);hold on
end



for j=1:5
fnamejs=['output/stats_D4_100_',num2str(j),'_t.json'];
val=jsondecode(fileread(fnamejs));

score=val.pred;
target=val.true;
[prec, tpr, fpr, thresh]=prec_rec(score, target);
%figure;
%plot(prec,tpr); hold on;
aupr_cnn(j)=abs(trapz(tpr,prec));
auROC_cnn(j)=abs(trapz(fpr,tpr));
figure(j)
plot(fpr,tpr);hold on
end


for j=1:5
fnamejs=['output/results_100_75_adam_',num2str(j),'.json'];
val=jsondecode(fileread(fnamejs));

score=val.pred;
target=val.true;
[prec, tpr, fpr, thresh]=prec_rec(score, target);
%figure;
%plot(prec,tpr); hold on;
aupr_AEnn(j)=abs(trapz(tpr,prec));
auROC_AEnn(j)=abs(trapz(fpr,tpr));
figure(j)
plot(fpr,tpr);hold on
end

figure(j)
    title('ROC curves for CNN, RNN, and AE-NN architecture')


close all;

aupr_DREAM4_winner=[0.536,0.377,0.390,0.349,0.213];
aupr_DREAM4_second=[0.512,0.396,0.380,0.372,0.178];
aupr_DREAM4_third=[0.490,0.327,0.326,0.4,0.159];


auROC_DREAM4_winner=[0.914,0.801,0.833,0.842,0.759];
auROC_DREAM4_second=[0.908,0.797,0.829,0.844,0.763];
auROC_DREAM4_third=[0.870,0.773,0.844,0.827,0.758];

figure;
set(0,'defaultAxesFontSize',9) 
b=bar([aupr_cnn;aupr_AEnn;aupr_rnn;aupr_DREAM4_winner;aupr_DREAM4_second;aupr_DREAM4_third]');
set(gca,'XTickLabel',{'Network1','Network2','Network3','Network4','Network5'})
l=legend('CNN','AE-NN','RNN','Pinna et al.','DREAM4 $2^{nd}$','DREAM4 $3^{rd}$');
l.Box='off';
title('Area under precision recall curve')
set(gca,'box','off')
xlim([0.5 5.5])
SaveImagePdf(gcf,4.1,4.1,'AUPR.pdf','pdf','in')
SaveImagePdfP(gcf,4.1,4.1,'AUPR1.pdf',[-.27 -.25 4.6 4.5],'in')

figure;
b=bar([auROC_cnn;auROC_AEnn;auroc_rnn;auROC_DREAM4_winner;auROC_DREAM4_second;auROC_DREAM4_third]');
set(gca,'XTickLabel',{'Network1','Network2','Network3','Network4','Network5'})
l=legend('CNN','AE-NN','RNN','Pinna et al.','DREAM4 $2^{nd}$','DREAM4 $3^{rd}$');
l.Box='off';
xlim([0.5 5.5])
title('Area under ROC curve')
set(gca,'box','off')
SaveImagePdf(gcf,4.5,4.5,'AUROC.pdf','pdf','in')
SaveImagePdfP(gcf,4.1,4.1,'AUROC1.pdf',[-.27 -.25 4.4 4.4],'in')

