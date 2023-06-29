clear all;
close all;
clc
%C:\MasterArbeit\WanchangMasterArbeit -20210129\Matlab\CEsBUExample1_2RVs_lnC_m
addpath('C:\MasterArbeit\WanchangMasterArbeit -20210129\Matlab\CEsBU')
%% Plot figure size definition
figsize = [10,10,600,480;
           10,10,800,600];
asize = 19;
%% Problem definition
global a0 DS ac T_final
dim = 2;
% fixed parameters
a0_initial = 2; % mm
ac = 50;% mm
T_final = 50;
a0 = a0_initial ;DS = 60;
T_final_text = ['T_{final}= ',num2str(T_final,'%g ')];

% RVs_prior distribution
% lnC, m~ multinormal(mu = [-33;3.5] si = [0.47,0.3],correlation coefficient = -0.9)
% Units are N and mm
% lambda_a0 = 1; mu_a0 = 1/lambda_a0; si_a0 = 1/lambda_a0; % lambda_a0 > 0
% mu_DS = 60; si_DS = 10;

mu_lnC = -33; si_lnC = 0.47;
mu_m = 3.5; si_m = 0.3;
prior_name = {'lnC','m'};
prior_distribution = {'normal','normal'};
mu_prior = [mu_lnC;mu_m];
si_prior = [si_lnC;si_m];
rho_lnC_m = -0.9;

lnC = ERADist('normal','MOM',[mu_lnC,si_lnC]);% -35.3 to -30.7
m = ERADist('normal','MOM',[mu_m,si_m]);% 2 to 5
% std matrix D
D = [si_lnC, 0;
    0,  si_m];
% Correlation matrix
R_XX = [1,rho_lnC_m;
    rho_lnC_m,1];
% Covariance matrix
COV = D*R_XX*D;
dist_prior = ERANataf([lnC,m],R_XX);
%% artificial data
t_m = 1:T_final;M = size(t_m,2);
x_given = [mu_lnC-si_lnC;mu_m-si_m];
a_true = a(x_given,t_m);
si = a_true(1)/300;
a_m = normrnd(a_true,si);
%% Plot a_true a_m
figure
set(gcf,'position',[figsize(1,:)])
p1 = plot(t_m,a_true,'c','LineWidth',2,'DisplayName','a_{True}');
hold on;
p2 = plot(t_m,a_m,'ro','MarkerFaceColor','r','DisplayName','a_{Meassurement}');
annotation('textbox',[0.15,0.6,0.5,0.2],'String','a = a(lnC,m,t)','FontSize',asize,'EdgeColor','none');
xlabel('t_m (year)');
ylabel('a CrackLength (mm)');
legend([p1,p2]);
s = ['a_{True} v.s. a_{M}'];
title(s)
name = ['figures\\CrackLength\\a_True and a_M'];
saveas(gcf,name,'png')

%% Likelihood definition
mu_error = 0; si_error = a_m(1)/10;
log_Likeli = @(x,year) log_Likelihood(x,year,a_m,t_m,mu_error,si_error);

%% parameters to choose
% CEsBU
Nstep = 1000;
Nlast = Nstep;
cvtarget = 1.5;
maxsteps = 100;
model = 'normal'; %choose 'normal' or 'vMFN'
k = 1;

%% CEsBU T_final solution
tic;
[Z_CEsBU1,lv,u_tot,x_tot,x_final,beta_tot,nu_tot,nESS,lnLeval_allyear] = CEsBU(Nstep, Nlast,log_Likeli,T_final, dist_prior, maxsteps, cvtarget,dim, k, model);
time_CEsBU1 = toc;
fprintf(['\nCEsBU1 took %fs\n'],time_CEsBU1)
%% analyse the model output crack length a
a_CEsBU1 = zeros(T_final,Nstep);
for t = 1: T_final
    a_CEsBU1(t,:) = a(x_final,t);
end
% 95% credible interval
mu_a_CEsBU1 = mean(a_CEsBU1,2); % T_final x1
si_a_CEsBU1 = std(a_CEsBU1,0,2);
alpha = 0.05;
lower_a_CEsBU1 = mu_a_CEsBU1 - si_a_CEsBU1*norminv(1-alpha/2);
upper_a_CEsBU1 = mu_a_CEsBU1 + si_a_CEsBU1*norminv(1-alpha/2);
%% Analyse the posterior distribution in T_final after CEsBU1
mu_CEsBU1 = zeros(dim,1);
si_CEsBU1 = zeros(dim,1);
for d = 1: dim
    mu_CEsBU1(d) = mean(x_final(d,:));
    si_CEsBU1(d) = std(x_final(d,:));
end
nESS_CEsBU1_text = ['nESS_{final_{CEsBU1}}=  ', num2str(nESS(end),'%.3f')];
Z_CEsBU1_text = ['Z_{final_{CEsBU1}}=  ', num2str(Z_CEsBU1,'%.3e')];
mu_prior_text = {};mu_CEsBU1_text = {};data_given_text = {};
si_prior_text = {};si_CEsBU1_text = {};
for d = 1:dim
    mu_prior_text{end+1} = ['mu_{prior}=  ', num2str(mu_prior(d),'%.3f')];
    si_prior_text{end+1} = [' si_{prior}=  ', num2str(si_prior(d),'%.3f')];
    data_given_text{end+1} = ['x_{given}= ', num2str(x_given(d),'%.3f')];
    mu_CEsBU1_text{end+1} = ['mu_{CEsBU1}=  ', num2str(mu_CEsBU1(d),'%.3f')];
    si_CEsBU1_text{end+1} = [' si_{CEsBU1}=  ', num2str(si_CEsBU1(d),'%.3f')];
end

%% Plot a_True,a_M and a_CEsBU1
figure
set(gcf,'position',figsize(1,:))
xxx = [t_m, fliplr(t_m)];
p1 = plot(t_m,a_true,'c','LineWidth',2,'DisplayName','a_{True}');
hold on;
p2 = plot(t_m,a_m,'ro','MarkerFaceColor','r','DisplayName','a_{Meassurements}');
p3 = plot(t_m,mu_a_CEsBU1,'m+-','DisplayName','mean_{a_{CEsBU1}}');
inBetween = [lower_a_CEsBU1', fliplr(upper_a_CEsBU1')];
p4 = fill(xxx, inBetween,'m','DisplayName','95%C.I.{a_{CEsBU1}}');
set(p4,'facealpha',0.05,'edgecolor','m','linestyle',':');
legend([p1,p2,p3,p4]);
annotation('textbox',[0.15,0.7,0.5,0.2],'String',{T_final_text;Z_CEsBU1_text;nESS_CEsBU1_text},'FontSize',asize,'EdgeColor','none');
xlabel('t_m (year)');ylabel('a CrackLength (mm)');
title(['a_{True},a_{M},a_{CEsBU1}'])
saveas(gcf,['figures\\CrackLength\\a_True,a_M and a_CEsBU1 '],'png')
%% Plot the interdist.for CEsBU1

figure
set(gcf,'position',figsize(2,:))
t = tiledlayout(1,2);
ax = {};
for d = 1: dim
    ax{end+1} = nexttile;
    % Intermediate samples plotting
    p1 = histogram(x_tot{1}(d,:),'Normalization','pdf','DisplayName','dist_{prior}');
    hold on;
    for l = 2: lv-1
        histogram(x_tot{l}(d,:),'Normalization','pdf')
        hold on;
    end
    p2 = histogram(x_tot{lv}(d,:),'Normalization','pdf','DisplayName','dist_{T_{final}}');
    p3 = plot(mu_prior(d),0,'r*','DisplayName','mu_{prior}');
    p4 = plot(x_given(d),0,'r+','DisplayName','x_{given}');
    p5 = plot(mu_CEsBU1(d),0,'ro','DisplayName','mu_{CEsBU1}');
    s = {mu_prior_text{d};data_given_text{d};mu_CEsBU1_text{d}};
    text(0.6,0.9,s,'FontSize',12,'Units','normalized');
    xlabel(['RV ',prior_name{d}]);
    ylabel('PDF')
    
end
sgtitle(['CEsBU1 in X space for all RVs ',T_final_text]);
legend(ax{1},[p1,p2,p3,p4,p5],'Location','northwestoutside');
t.TileSpacing = 'compact';
t.Padding = 'compact';
saveas(gcf,['figures\\InterDist\\CEsBU1 in X space for all RVs',T_final_text],'png');
T_object_list = [5,10,15,20,25,30,35,40,45];
for i = 1:length(T_object_list)
    T_object = T_object_list(i)
    T_object_text = [' T_{object}= ',num2str(T_object)];
    mu_a_CEsBU1_object_text = ['mu_{a_{CEsBU1_{object}}}: ',num2str(mu_a_CEsBU1(T_object))];
    si_a_CEsBU1_object_text = ['si_{a_{CEsBU1_{object}}}: ',num2str(si_a_CEsBU1(T_object))];
    %% perform the same procedure for T_object
    [Z_CEsBU2,lv_object,u_tot_object,x_tot_object,x_object,beta_tot_object,nu_tot_object,nESS_object,lnLeval_allyear_object] = CEsBU(Nstep, Nlast,log_Likeli,T_object, dist_prior, maxsteps, cvtarget,dim, k, model);
    
    %% output samples
    a_CEsBU2 = zeros(T_final,Nstep);
    mu_a_CEsBU2 = zeros(T_final,1);
    si_a_CEsBU2 = zeros(T_final,1);
    for t = 1: T_final
        a_CEsBU2(t,:) = a(x_object,t);
        mu_a_CEsBU2(t) = mean(a_CEsBU2(t,:));
        si_a_CEsBU2(t) = std(a_CEsBU2(t,:));
    end
    
    % 95% credible interval
    alpha = 0.05;
    lower_a_CEsBU2 = mu_a_CEsBU2 - si_a_CEsBU2*norminv(1-alpha/2);
    upper_a_CEsBU2 = mu_a_CEsBU2 + si_a_CEsBU2*norminv(1-alpha/2);
    
    %% Posterior distribution in T_object
    mu_CEsBU2 = zeros(dim,1);
    si_CEsBU2 = zeros(dim,1);
    for d = 1: dim
        mu_CEsBU2(d) = mean(x_object(d,:));
        si_CEsBU2(d) = std(x_object(d,:));
    end
    %% Text
    nESS_CEsBU2_text = ['nESS_{object_{CEsBU2}}=  ', num2str(nESS_object(end),'%.3f')];
    Z_CEsBU2_text = ['Z_{object_{CEsBU2}}=  ', num2str(Z_CEsBU2,'%.3e')];
    mu_CEsBU2_text = {};
    si_CEsBU2_text = {};
    for d = 1:dim
        mu_CEsBU2_text{end+1} = ['mu_{CEsBU2}:  ', num2str(mu_CEsBU2(d),'%.3f')];
        si_CEsBU2_text{end+1} = ['si_{CEsBU2}:  ', num2str(si_CEsBU2(d),'%.3f')];
    end
    mu_a_CEsBU2_object_text = ['mu_{a_{CEsBU2_{object}}}: ',num2str(mu_a_CEsBU2(T_object),'%.3f')];
    si_a_CEsBU2_object_text = ['si_{a_{CEsBU2_{object}}}: ',num2str(si_a_CEsBU2(T_object),'%.3f')];
    %% plot 1 output plot
    figure
    set(gcf,'position',figsize(1,:))
    % Plot the 95% confidence interval
    p1 = plot(t_m,a_true,'c','LineWidth',2,'DisplayName','a_{True}');
    hold on;
    p2 = plot(t_m,a_m,'ro','MarkerFaceColor','r','DisplayName','a_{Meassurements}');
    p3 = plot(t_m,mu_a_CEsBU2,'k+-','DisplayName','mean_{a_{CEsBU2}}');
    xxx = [t_m, fliplr(t_m)];
    inBetween = [lower_a_CEsBU2', fliplr(upper_a_CEsBU2')];
    p4 = fill(xxx, inBetween, 'k','DisplayName','95%C.I.{a_{CEsBU2}}','linestyle',':');
    p5 = xline(T_object,'--k','DisplayName','T_{object}');
    set(p4,'facealpha',0.05);
    annotation('textbox',[0.15,0.7,0.5,0.2],'String',Z_CEsBU2_text,'FontSize',asize,'EdgeColor','none');
    legend([p1,p2,p3,p4,p5]);
    xlabel('t_m (year)');ylabel('a CrackLength (mm)');
    title(['a_{True},a_{M},a_{CEsBU2} where ',T_object_text])
    saveas(gcf,['figures\\CrackLength\\a_True,a_M,a_CEsBU2 with ',T_object_text],'png')
    
    %% plot 1 output plot
    xxx = [t_m, fliplr(t_m)];
    figure
    set(gcf,'position',figsize(1,:))
    % Plot the 95% confidence interval
    p1 = plot(t_m,a_true,'c','LineWidth',2,'DisplayName','a_{True}');
    hold on;
    p2 = plot(t_m,a_m,'ro','MarkerFaceColor','r','DisplayName','a_{Meassurements}');
    p3 = plot(t_m,mu_a_CEsBU1,'m+-','DisplayName','mean_{a_{CEsBU1}}');
    inBetween = [lower_a_CEsBU1', fliplr(upper_a_CEsBU1')];
    p4 = fill(xxx, inBetween,'m','DisplayName','95%C.I.{a_{CEsBU1}}');
    set(p4,'facealpha',0.05,'edgecolor','m','linestyle',':');
    
    p5 = plot(t_m,mu_a_CEsBU2,'k+-','DisplayName','mean_{a_{CEsBU2}}');
    inBetween = [lower_a_CEsBU2', fliplr(upper_a_CEsBU2')];
    p6 = fill(xxx, inBetween, 'k','DisplayName','95%C.I.{a_{CEsBU2}}');
    set(p6,'facealpha',0.05,'edgecolor','k','linestyle',':');
    p7 = xline(T_object,'--k','DisplayName','T_{object}');
    annotation('textbox',[0.15,0.7,0.5,0.2],'String',{Z_CEsBU1_text;Z_CEsBU2_text},'FontSize',asize,'EdgeColor','none');
    legend([p1,p2,p3,p4,p5,p6,p7]);
    xlabel('t_m (year)');ylabel('a CrackLength (mm)');
    title(['a_{True},a_{M},a_{CEsBU1,2} with the C.I. ',T_object_text])
    saveas(gcf,['figures\\CrackLength\\a_True,a_M,a_CEsBU1,2 with ',T_object_text],'png')
    
    %% plot2 I plot for each RV
    figure
    set(gcf,'position',figsize(2,:))
    t = tiledlayout(1,2);
    ax = {};
    for d = 1: dim
        ax{end+1} = nexttile;
        % Intermediate samples plotting
        p1 = histogram(x_tot_object{1}(d,:),'Normalization','pdf','DisplayName','dist_{prior}');
        hold on;
        for l = 2: lv_object-1
            histogram(x_tot_object{l}(d,:),'Normalization','pdf')
        end
        p2 = histogram(x_tot_object{lv_object}(d,:),'Normalization','pdf','DisplayName','dist_{T_{final}}');
        p3 = plot(mu_prior(d),0,'r*','DisplayName','mu_{prior}');
        p4 = plot(x_given(d),0,'r+','DisplayName','x_{given}');
        p5 = plot(mu_CEsBU2(d),0,'ro','DisplayName','mu_{CEsBU1}');
        s = {mu_prior_text{d};data_given_text{d};mu_CEsBU2_text{d}};
        text(0.6,0.9,s,'FontSize',12,'Units','normalized');
        xlabel(['RV ',prior_name{d}]);
        ylabel('PDF')
    end
    sgtitle(['CEsBU2 in X space for all RVs ',T_object_text]);
    legend(ax{1},[p1,p2,p3,p4,p5],'Location','northwestoutside');
    t.TileSpacing = 'compact';
    t.Padding = 'compact';
    saveas(gcf,['figures\\InterDist\\CEsBU2 in X space for all RVs with ',T_object_text],'png');
    
    %% Sequential step 1
    [U_object1,X_object1,Z_object1] = Sequential_step1(dist_prior,T_object,lv,u_tot,nu_tot,lnLeval_allyear,dim, k, model,Nstep,nESS);
    %% Sequential step 2
    [U_object2,X_object2,Z_object2] = Sequential_step2(dist_prior,T_object,lv,u_tot,nu_tot,lnLeval_allyear,dim, k, model,Nstep);
    %% Sequential step 3
    [U_object3,X_object3,Z_object3] = Sequential_step3(dist_prior,T_object,lv,u_tot,nu_tot,lnLeval_allyear,dim, k, model,Nstep,nESS);
    %% Sequential step 4
    [U_object4,X_object4,Z_object4] = Sequential_step4(dist_prior,T_object,lv,u_tot,nu_tot,lnLeval_allyear,dim, k, model,Nstep);
   
    %% Sequential step output
    mu_SS1 = zeros(dim,1);
    si_SS1 = zeros(dim,1);
    mu_SS2 = zeros(dim,1);
    si_SS2 = zeros(dim,1);
    mu_SS3 = zeros(dim,1);
    si_SS3 = zeros(dim,1);        
    mu_SS4 = zeros(dim,1);
    si_SS4 = zeros(dim,1);
    for d = 1: dim
        mu_SS1(d) = mean(X_object1(d,:));
        si_SS1(d) = std(X_object1(d,:));
        mu_SS2(d) = mean(X_object2(d,:));
        si_SS2(d) = std(X_object2(d,:));
        mu_SS3(d) = mean(X_object3(d,:));
        si_SS3(d) = std(X_object3(d,:));
        mu_SS4(d) = mean(X_object4(d,:));
        si_SS4(d) = std(X_object4(d,:));
    end
    a_SS1 = zeros(T_final,Nstep);
    a_SS2 = zeros(T_final,Nstep);
    a_SS3 = zeros(T_final,Nstep);
    a_SS4 = zeros(T_final,Nstep);
    for t = 1:T_final
        a_SS1(t,:) = a(X_object1,t);
        a_SS2(t,:) = a(X_object2,t);
        a_SS3(t,:) = a(X_object3,t);
        a_SS4(t,:) = a(X_object4,t);
    end
    mu_a_SS1 = mean(a_SS1,2);
    mu_a_SS2 = mean(a_SS2,2);
    mu_a_SS3 = mean(a_SS3,2);
    mu_a_SS4 = mean(a_SS4,2);
    
    si_a_SS1 = std(a_SS1,0,2);
    si_a_SS2 = std(a_SS2,0,2);
    si_a_SS3 = std(a_SS3,0,2);
    si_a_SS4 = std(a_SS4,0,2);
    % credible interval
    alpha = 0.05;
    lower_a_SS1 = mu_a_SS1 - si_a_SS1 * norminv(1-alpha/2);
    upper_a_SS1 = mu_a_SS1 + si_a_SS1 * norminv(1-alpha/2);
    lower_a_SS2 = mu_a_SS2 - si_a_SS2 * norminv(1-alpha/2);
    upper_a_SS2 = mu_a_SS2 + si_a_SS2 * norminv(1-alpha/2);
    lower_a_SS3 = mu_a_SS3 - si_a_SS3 * norminv(1-alpha/2);
    upper_a_SS3 = mu_a_SS3 + si_a_SS3 * norminv(1-alpha/2);
    lower_a_SS4 = mu_a_SS4 - si_a_SS4 * norminv(1-alpha/2);
    upper_a_SS4 = mu_a_SS4 + si_a_SS4 * norminv(1-alpha/2);
    %% Text for figure plot and comparision
    Z_SS1_text = ['Z_{object_{SS1}}=  ', num2str(Z_object1,'%.3e')];
    Z_SS2_text = ['Z_{object_{SS2}}=  ', num2str(Z_object2,'%.3e')];
    Z_SS3_text = ['Z_{object_{SS3}}=  ', num2str(Z_object3,'%.3e')];
    Z_SS4_text = ['Z_{object_{SS4}}=  ', num2str(Z_object4,'%.3e')];
    mu_SS1_text = {};mu_SS2_text = {};mu_SS3_text = {};mu_SS4_text = {};
    si_SS1_text = {};si_SS2_text = {};si_SS3_text = {};si_SS4_text = {};
    for d = 1:dim
        mu_SS1_text{end+1} = ['mu_{SS1}=  ', num2str(mu_SS1(d),'%.3f')];
        si_SS1_text{end+1} = ['si_{SS1}=  ', num2str(si_SS1(d),'%.3f')];
        mu_SS2_text{end+1} = ['mu_{SS2}=  ', num2str(mu_SS2(d),'%.3f')];
        si_SS2_text{end+1} = ['si_{SS2}=  ', num2str(si_SS2(d),'%.3f')];
        mu_SS3_text{end+1} = ['mu_{SS3}=  ', num2str(mu_SS3(d),'%.3f')];
        si_SS3_text{end+1} = ['si_{SS3}=  ', num2str(si_SS3(d),'%.3f')];
        mu_SS4_text{end+1} = ['mu_{SS4}=  ', num2str(mu_SS4(d),'%.3f')];
        si_SS4_text{end+1} = ['si_{SS4}=  ', num2str(si_SS4(d),'%.3f')];
    end
    mu_a_SS1_object_text = ['mu_{a_{SS1_{object}}}:',num2str(mu_a_SS1(T_object),'%.3f')];
    mu_a_SS2_object_text = ['mu_{a_{SS2_{object}}}:',num2str(mu_a_SS2(T_object),'%.3f')];
    mu_a_SS3_object_text = ['mu_{a_{SS3_{object}}}:',num2str(mu_a_SS3(T_object),'%.3f')];
    mu_a_SS4_object_text = ['mu_{a_{SS4_{object}}}:',num2str(mu_a_SS4(T_object),'%.3f')];
    si_a_SS1_object_text = ['si_{a_{SS1_{object}}}:',num2str(si_a_SS1(T_object),'%.3f')];
    si_a_SS2_object_text = ['si_{a_{SS2_{object}}}:',num2str(si_a_SS2(T_object),'%.3f')];
    si_a_SS3_object_text = ['si_{a_{SS3_{object}}}:',num2str(si_a_SS3(T_object),'%.3f')];
    si_a_SS4_object_text = ['si_{a_{SS4_{object}}}:',num2str(si_a_SS4(T_object),'%.3f')];
    %% Plot a_true; a_m; a_SS1
    figure
    % figure size setting
    set(gcf,'position',figsize(1,:))
    % Plot the 95% confidence interval
    p1 = plot(t_m,a_true,'c','LineWidth',2,'DisplayName','a_{True}');
    hold on;
    p2 = plot(t_m,a_m,'ro','MarkerFaceColor','r','DisplayName','a_{Meassurements}');
    p3 = plot(t_m,mu_a_SS1,'b+','DisplayName','mean_{a_{SS1}}');
    xxx = [t_m, fliplr(t_m)];
    inBetween = [lower_a_SS1', fliplr(upper_a_SS1')];
    p4 = fill(xxx, inBetween, 'b','DisplayName','95%C.I.{a_{SS1}}');
    set(p4,'facealpha',0.05,'edgecolor','b');
    p5 = xline(T_object,'--k','DisplayName','T_{object}');
    annotation('textbox',[0.15,0.7,0.5,0.2],'String',{Z_SS1_text},'FontSize',asize,'EdgeColor','none');
    legend([p1,p2,p3,p4,p5]);
    xlabel('t_m (year)');ylabel('a CrackLength (mm)')
    title(['a_{True},a_{M},a_{SS1} with ',T_object_text])
    saveas(gcf,['figures\\CrackLength\\a_True,a_M,a_SS1 with ', T_object_text],'png')
    
    %% Plot a_true; a_m; a_SS2
    figure
    % figure size setting
    set(gcf,'position',figsize(1,:))
    p1 = plot(t_m,a_true,'c','LineWidth',2,'DisplayName','a_{True}');
    hold on;
    p2 = plot(t_m,a_m,'ro','MarkerFaceColor','r','DisplayName','a_{Meassurements}');
    p3 = plot(t_m,mu_a_SS2,'g+','DisplayName','mean_{a_{SS2}}');
    hold on;
    xxx = [t_m, fliplr(t_m)];
    inBetween = [lower_a_SS2', fliplr(upper_a_SS2')];
    p4 = fill(xxx, inBetween, 'g','DisplayName','95%C.I.{a_{SS2}}');
    set(p4,'facealpha',0.05,'edgecolor','g');
    p5 = xline(T_object,'--k','DisplayName','T_{object}');
    annotation('textbox',[0.15,0.7,0.5,0.2],'String',{Z_SS2_text},'FontSize',asize,'EdgeColor','none');
    legend([p1,p2,p3,p4,p5]);
    xlabel('t_m (year)');ylabel('a CrackLength (mm)');
    title(['a_{True},a_{M},a_{SS2} with ',T_object_text])
    saveas(gcf,['figures\\CrackLength\\a_True,a_M,a_SS2 with ',T_object_text],'png')
    
    %% Plot a_true; a_m; a_SS3
    figure
    % figure size setting
    set(gcf,'position',figsize(1,:))
    p1 = plot(t_m,a_true,'c','LineWidth',2,'DisplayName','a_{True}');
    hold on;
    p2 = plot(t_m,a_m,'ro','MarkerFaceColor','r','DisplayName','a_{Meassurements}');
    p3 = plot(t_m,mu_a_SS3,'y+','DisplayName','mean_{a_{SS3}}');
    xxx = [t_m, fliplr(t_m)];
    inBetween = [lower_a_SS3', fliplr(upper_a_SS3')];
    p4 = fill(xxx, inBetween, 'y','DisplayName','95%C.I.{a_{SS3}}');
    set(p4,'facealpha',0.05,'edgecolor','y');
    p5 = xline(T_object,'--k','DisplayName','T_{object}');
    annotation('textbox',[0.15,0.7,0.5,0.2],'String',{Z_SS3_text},'FontSize',asize,'EdgeColor','none');
    legend([p1,p2,p3,p4,p5]);
    xlabel('t_m (year)');ylabel('a CrackLength (mm)')
    title(['a_{True},a_{M},a_{SS3} with ',T_object_text])
    saveas(gcf,['figures\\CrackLength\\a_True,a_M,a_SS3 with ',T_object_text],'png')
    %% Plot a_true; a_m; a_SS4
    figure
    % figure size setting
    set(gcf,'position',figsize(1,:))
    p1 = plot(t_m,a_true,'c','LineWidth',2,'DisplayName','a_{True}');
    hold on;
    p2 = plot(t_m,a_m,'ro','MarkerFaceColor','r','DisplayName','a_{Meassurements}');
    p3 = plot(t_m,mu_a_SS4,'r+','DisplayName','mean_{a_{SS4}}');
    xxx = [t_m, fliplr(t_m)];
    inBetween = [lower_a_SS4', fliplr(upper_a_SS4')];
    p4 = fill(xxx, inBetween, 'r','DisplayName','95%C.I.{a_{SS4}}');
    set(p4,'facealpha',0.05,'edgecolor','r');
    p5 = xline(T_object,'--k','DisplayName','T_{object}');
    annotation('textbox',[0.15,0.7,0.5,0.2],'String',{Z_SS4_text},'FontSize',asize,'EdgeColor','none');
    legend([p1,p2,p3,p4,p5]);
    xlabel('t_m (year)');ylabel('a CrackLength (mm)')
    title(['a_{True},a_{M},a_{SS4} with ',T_object_text])
    saveas(gcf,['figures\\CrackLength\\a_True,a_M,a_SS4 with ',T_object_text],'png')
    
    %% Plot a_true; a_m; a_SS1,2,3,4
    figure
    % figure size setting
    set(gcf,'position',figsize(2,:))
    p1 = plot(t_m,a_true,'c','LineWidth',2,'DisplayName','a_{True}');
    hold on;
    xxx = [t_m, fliplr(t_m)];
    p2 = plot(t_m,a_m,'ro','MarkerFaceColor','r','DisplayName','a_{Meassurements}');
    p3 = plot(t_m,mu_a_SS1,'b+','DisplayName','mean_{a_{SS1}}');
    inBetween = [lower_a_SS1', fliplr(upper_a_SS1')];
    p4 = fill(xxx, inBetween, 'b','DisplayName','95%C.I.{a_{SS1}}');
    set(p4,'facealpha',0.1);set(p4,'edgecolor','b');
    p5 = plot(t_m,mu_a_SS2,'g+','DisplayName','mean_{a_{SS2}}');
    inBetween = [lower_a_SS2', fliplr(upper_a_SS2')];
    p6 = fill(xxx, inBetween, 'g','DisplayName','95%C.I.{a_{SS2}}');
    set(p6,'facealpha',0.1);set(p6,'edgecolor','g');
    p7 = plot(t_m,mu_a_SS3,'y+','DisplayName','mean_{a_{SS3}}');
    inBetween = [lower_a_SS3', fliplr(upper_a_SS3')];
    p8 = fill(xxx, inBetween, 'y','DisplayName','95%C.I.{a_{SS3}}');
    set(p8,'facealpha',0.1,'edgecolor','y');
    
    p9 = plot(t_m,mu_a_SS4,'r+','DisplayName','mean_{a_{SS4}}');
    inBetween = [lower_a_SS4', fliplr(upper_a_SS4')];
    p10 = fill(xxx, inBetween, 'r','DisplayName','95%C.I.{a_{SS4}}');
    set(p10,'facealpha',0.1,'edgecolor','r');
    
    s1 = {Z_CEsBU1_text;Z_CEsBU2_text;Z_SS1_text;Z_SS2_text;Z_SS3_text;Z_SS4_text};
    text(0.05,0.6,s1,'FontSize',asize,'Units','normalized');
    p11 = xline(T_object,'--k','DisplayName','T_{object}');
    
    %legend([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11],'Location','bestoutside');
    xlabel('t_m (year)');ylabel('a CrackLength (mm)');
    title(['a_{True},a_{M},a_{SS1,2,3,4} with ',T_object_text])
    saveas(gcf,['figures\\CrackLength\\a_True,a_M,a_SS1,2,3,4 with ',T_object_text],'png')
    %% Plot a_true; a_m; a_CEsBU1,2; a_SS1,2,3,4
    figure
    % figure size setting
    set(gcf,'position',figsize(2,:))
    p1 = plot(t_m,a_true,'c','LineWidth',2,'DisplayName','a_{True}');
    hold on;
    xxx = [t_m, fliplr(t_m)];
    p2 = plot(t_m,a_m,'ro','MarkerFaceColor','r','DisplayName','a_{Meassurements}');
    p3 = plot(t_m,mu_a_CEsBU1,'m+-','DisplayName','mean_{a_{CEsBU1}}');
    inBetween = [lower_a_CEsBU1', fliplr(upper_a_CEsBU1')];
    p4 = fill(xxx, inBetween, 'm','DisplayName','95%C.I.{a_{CEsBU1}}');
    set(p4,'facealpha',0.1,'edgecolor','m');
    p5 = plot(t_m,mu_a_CEsBU2,'k+-','DisplayName','mean_{a_{CEsBU2}}');
    inBetween = [lower_a_CEsBU2', fliplr(upper_a_CEsBU2')];
    p6 = fill(xxx, inBetween, 'k','DisplayName','95%C.I.{a_{CEsBU2}}');
    set(p6,'facealpha',0.1,'edgecolor','k');
    p7 = plot(t_m,mu_a_SS1,'b+','DisplayName','mean_{a_{SS1}}');
    inBetween = [lower_a_SS1', fliplr(upper_a_SS1')];
    p8 = fill(xxx, inBetween, 'b','DisplayName','95%C.I.{a_{SS1}}');
    set(p8,'facealpha',0.1,'edgecolor','b');
    p9 = plot(t_m,mu_a_SS2,'g+','DisplayName','mean_{a_{SS2}}');
    inBetween = [lower_a_SS2', fliplr(upper_a_SS2')];
    p10 = fill(xxx, inBetween, 'g','DisplayName','95%C.I.{a_{SS2}}');
    set(p10,'facealpha',0.1,'edgecolor','g');    
    p11 = plot(t_m,mu_a_SS3,'y+','DisplayName','mean_{a_{SS3}}');
    inBetween = [lower_a_SS3', fliplr(upper_a_SS3')];
    p12 = fill(xxx, inBetween, 'y','DisplayName','95%C.I.{a_{SS3}}');
    set(p12,'facealpha',0.1,'edgecolor','y');
    p13 = plot(t_m,mu_a_SS4,'r+','DisplayName','mean_{a_{SS4}}');
    inBetween = [lower_a_SS4', fliplr(upper_a_SS4')];
    p14 = fill(xxx, inBetween, 'r','DisplayName','95%C.I.{a_{SS4}}');
    set(p14,'facealpha',0.1,'edgecolor','r');    
    s1 = {Z_CEsBU1_text;Z_CEsBU2_text;Z_SS1_text;Z_SS2_text;Z_SS3_text;Z_SS4_text}%mu_a_CEsBU1_object_text;mu_a_CEsBU2_object_text;mu_a_SS1_object_text;mu_a_SS2_object_text,mu_a_SS3_object_text};
    
    text(0.05,0.6,s1,'FontSize',asize,'Units','normalized')
    p15 = xline(T_object,'--k','DisplayName','T_{object}');    
    %legend([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15],'Location','bestoutside');
    xlabel('t_m (year)');ylabel('a CrackLength (mm)');
    title(['a_{True},a_{M},a_{CEsBU1,2},a_{SS1,2,3,4} with ',T_object_text])
    saveas(gcf,['figures\\CrackLength\\a_True,a_M,a_CEsBU1,2,a_SS1,2,3,4 with ',T_object_text],'png')
    
    
    %% plot the dist.for Prior; CEsBU1 ;SS1,2,3
    figure;
    % figure size setting
    set(gcf,'position',figsize(2,:))
    t = tiledlayout(1,2);
    ax = {};
    for d = 1: dim
        ax{end+1} = nexttile;
        % Intermediate samples plotting
        p1 = histogram(x_tot{1}(d,:),'Normalization','pdf','DisplayName','dist._{Prior}');
        hold on;
        for l = 2: lv-1
            histogram(x_tot{l}(d,:),'Normalization','pdf');
        end
        p2 = histogram(x_tot{lv}(d,:),'Normalization','pdf','DisplayName','dist_{T_{final}CEsBU1}');
        p3 = histogram(X_object1(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}SS1}');
        p4 = histogram(X_object2(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}SS2}');
        p5 = histogram(X_object3(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}SS3}');
        p6 = plot(mu_prior(d),0,'r*','DisplayName','mu_{prior}');
        p7 = plot(x_given(d),0,'r+','DisplayName','x_{given}');
        p8 = plot(mu_CEsBU1(d),0,'rx','DisplayName','mu_{T_{final}CEsBU}');
        p9 = plot(mu_SS1(d),0,'gd','DisplayName','mu_{T_{object}SS1}');
        p10 = plot(mu_SS2(d),0,'gs','DisplayName','mu_{T_{object}SS2}');
        p11 = plot(mu_SS3(d),0,'gv','DisplayName','mu_{T_{object}SS3}');
        p12 = plot(mu_SS4(d),0,'go','DisplayName','mu_{T_{object}SS4}');
        xlabel(['RV ',prior_name{d}]); ylabel('PDF');
        s = {mu_prior_text{d},data_given_text{d},mu_CEsBU1_text{d},mu_SS1_text{d},mu_SS2_text{d},mu_SS3_text{d},mu_SS4_text{d}};
        text(0.6,0.8,s,'FontSize',12,'Units','normalized');
    end
    sgtitle(['CEsBU1 with samples from SS1,2,3,4 in X space for all RVs ',T_object_text])
    legend(ax{1},[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12],'Location','northwestoutside')
    t.TileSpacing = 'compact';
    t.Padding = 'compact';
    name= ['figures\\InterDist\\CEsBU1 with samples from SS1,2,3,4 in X space for all RVs ', T_object_text];
    saveas(gcf,name,'png')
    %% plot the dist.for Prior; CEsBU2 ;SS1,2,3,4
    figure;
    % figure size setting
    set(gcf,'position',figsize(2,:))
    t = tiledlayout(1,2);
    ax = {};
    for d = 1:dim
        ax{end+1} = nexttile;
        p1 = histogram(x_tot_object{1}(d,:),'Normalization','pdf','DisplayName','dist_{Prior}');
        hold on;
        p2 = histogram(x_object(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}CEsBU2}');
        p3 = histogram(X_object1(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}SS1}');
        p4 = histogram(X_object2(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}SS2}');
        p5 = histogram(X_object3(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}SS3}');
        p6 = plot(mu_prior(d),0,'r+','DisplayName','mu_{prior}');
        p7 = plot(x_given(d),0,'r*','DisplayName','x_{given}');
        p8 = plot(mu_CEsBU2(d),0,'ro','DisplayName','mu_{T_{object}CEsBU}');
        p9 = plot(mu_SS1(d),0,'gd','DisplayName','mu_{T_{object}SS1}');
        p10 = plot(mu_SS2(d),0,'gs','DisplayName','mu_{T_{object}SS2}');
        p11 = plot(mu_SS3(d),0,'gv','DisplayName','mu_{T_{object}SS3}');
        p12 = plot(mu_SS4(d),0,'go','DisplayName','mu_{T_{object}SS4}');
        xlabel(['RV ',prior_name{d}]);  ylabel('PDF');
        s = {mu_prior_text{d},data_given_text{d},mu_CEsBU2_text{d},mu_SS1_text{d},mu_SS2_text{d},mu_SS3_text{d},mu_SS4_text{d}};
        text(0.6,0.8,s,'FontSize',12,'Units','normalized');
    end
    sgtitle(['CEsBU2 with samples from SS1,2,3,4 in X space for all RVs ',T_object_text])
    % add legend
    legend(ax{1},[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12],'Location','northwestoutside')
    t.TileSpacing = 'compact';
    t.Padding = 'compact';
    name = ['figures\\InterDist\\CEsBU2 with samples from SS1,2,3,4 in X space for all RVs ',T_object_text];
    saveas(gcf,name,'png')
    
    %% plot the dist.for Prior; CEsBU1,2 ;SS1,2,3,4
    figure;
    % figure size setting
    set(gcf,'position',figsize(2,:))
    t = tiledlayout(1,2);
    ax = {};
    for d = 1:dim
        ax{end+1}= nexttile
        p1 = histogram(x_tot_object{1}(d,:),'Normalization','pdf','DisplayName','dist_{Prior}');
        hold on;
        p2 = histogram(x_final(d,:),'Normalization','pdf','DisplayName','dist_{T_{final}CEsBU1}');
        p3 = histogram(x_object(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}CEsBU2}');
        p4 = histogram(X_object1(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}SS1}');
        p5 = histogram(X_object2(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}SS2}');
        p6 = histogram(X_object3(d,:),'Normalization','pdf','DisplayName','dist_{T_{object}SS3}');
        p7 = plot(mu_prior(d),0,'r+','DisplayName','mu_{prior}');
        p8 = plot(x_given(d),0,'r*','DisplayName','x_{given}');
        p9 = plot(mu_CEsBU1(d),0,'rx','DisplayName','mu_{T_{final}CEsBU}');
        p10 = plot(mu_CEsBU2(d),0,'ro','DisplayName','mu_{T_{object}CEsBU}');
        p11 = plot(mu_SS1(d),0,'gd','DisplayName','mu_{T_{object}SS1}');
        p12 = plot(mu_SS2(d),0,'gs','DisplayName','mu_{T_{object}SS2}');
        p13 = plot(mu_SS3(d),0,'gv','DisplayName','mu_{T_{object}SS3}');
        p14 = plot(mu_SS4(d),0,'go','DisplayName','mu_{T_{object}SS4}');
        xlabel(['RV ',prior_name{d}]);  ylabel('PDF')
        s = {mu_prior_text{d},data_given_text{d},mu_CEsBU1_text{d},mu_CEsBU2_text{d},mu_SS1_text{d},mu_SS2_text{d},mu_SS3_text{d},mu_SS4_text{d}};
        text(0.6,0.8,s,'FontSize',12,'Units','normalized');
    end
    sgtitle(['CEsBU1,2 and SS1,2,3,4 in X space for all RVs ',T_object_text])
    legend(ax{1},[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14],'Location','northwestoutside')
    t.TileSpacing = 'compact';
    t.Padding = 'compact';
    name = ['figures\\InterDist\\Distribution for Prior CEsBU1,2 and SS1,2,3,4 in X space for all RVs ',T_object_text];
    saveas(gcf,name,'png')
end