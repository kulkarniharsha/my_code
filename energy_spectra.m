clc
clear all
data=xlsread('P-36-near-impeller.xlsx');
% data=xlsread('P-38-x_150mm_impeller.xlsx');
% data=xlsread('P-39-x_150mm_opposite.xlsx');
data(isnan(data))=0;
af=data(3:65534,2);
N=length(af);
% S=sum(data(:,2));
umean=1.0;%abs(S/N)%mean velocity at the measurement location
dt=0.0005;
T=N*dt;
Fs=1/dt;%sampling frequency
tspan=0:dt:T-dt;
sig=af;
Y=abs(fft(sig))/N;
nyquist=0.5*Fs;
Fmin=1/T;
f=linspace(Fmin,nyquist,N/2);
Y(1)=[];
YP=Y(1:N/2);
plot(f,YP)
pwr_fft=(YP.^2)/N;
wave_no=f.*(2*pi./umean);
ek=pwr_fft./(2*pi./umean);
figure
loglog(wave_no(1:N/2),ek(1:N/2))

