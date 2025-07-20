% Lab 8
%% Task 1 - Lowpass FIR/IIR comparison
fs=8000;
t=0:1/fs:1;
signal=sin(2*pi*500*t)+sin(2*pi*2000*t);
noise=0.5*randn(size(t));
signal=signal+noise;
filtered_FIR=filter(Num,1,signal);
filtered_IIR=filtfilt(SOS,G,signal);
figure;
subplot(311);plot(t(1:100),signal(1:100));grid on;
title('Original signal');xlabel('Time (s)');ylabel('Amplitude');

subplot(312);plot(t(1:100),filtered_FIR(1:100));grid on;
title('Filtered signal (FIR)');xlabel('Time (s)');ylabel('Amplitude');

subplot(313);plot(t(1:100),filtered_IIR(1:100));grid on;
title('Filtered signal (IIR)');xlabel('Time (s)');ylabel('Amplitude');

sgtitle('Comparison of FIR and IIR filtering');

%Compute FFT
N=length(signal);
f=linspace(0,fs/2,N/2);
X_original=abs(fft(signal));
X_FIR=abs(fft(filtered_FIR));
X_IIR=abs(fft(filtered_IIR));

%Normalize and keep only positive frequencies
X_original=X_original(1:N/2)/max(X_original);
X_FIR=X_FIR(1:N/2)/max(X_FIR);
X_IIR=X_IIR(1:N/2)/max(X_IIR);

figure;
subplot(311);plot(f,X_original);grid on;
title('Frequency response of original signal');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(312);plot(f,X_FIR);grid on;
title('Frequency response of filtered signal (FIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(313);plot(f,X_IIR);grid on;
title('Frequency response of filtered signal (IIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

sgtitle('Comparison of frequency responses');

%% Task 2 - Highpass FIR/IIR comparison
fs=8000;
t=0:1/fs:1;
signal=sin(2*pi*500*t)+sin(2*pi*2500*t);
noise=0.5*randn(size(t));
signal=signal+noise;
filtered_FIR=filter(Num1,1,signal);
filtered_IIR=filtfilt(SOS1,G1,signal);
figure;
subplot(311);plot(t(1:100),signal(1:100));grid on;
title('Original signal');xlabel('Time (s)');ylabel('Amplitude');

subplot(312);plot(t(1:100),filtered_FIR(1:100));grid on;
title('Filtered signal (FIR)');xlabel('Time (s)');ylabel('Amplitude');

subplot(313);plot(t(1:100),filtered_IIR(1:100));grid on;
title('Filtered signal (IIR)');xlabel('Time (s)');ylabel('Amplitude');

sgtitle('Comparison of FIR and IIR filtering');

%Compute FFT
N=length(signal);
f=linspace(0,fs/2,N/2);
X_original=abs(fft(signal));
X_FIR=abs(fft(filtered_FIR));
X_IIR=abs(fft(filtered_IIR));

%Normalize and keep only positive frequencies
X_original=X_original(1:N/2)/max(X_original);
X_FIR=X_FIR(1:N/2)/max(X_FIR);
X_IIR=X_IIR(1:N/2)/max(X_IIR);

figure;
subplot(311);plot(f,X_original);grid on;
title('Frequency response of original signal');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(312);plot(f,X_FIR);grid on;
title('Frequency response of filtered signal (FIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(313);plot(f,X_IIR);grid on;
title('Frequency response of filtered signal (IIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

sgtitle('Comparison of frequency responses');

%% Task 3 - Bandpass FIR/IIR comparison
fs=8000;
t=0:1/fs:1;
signal = sin(2*pi*100*t) + sin(2*pi*800*t) + sin(2*pi*370*t);
noise=0.5*randn(size(t));
signal=signal+noise;
filtered_FIR=filter(Num2,1,signal);
filtered_IIR=filtfilt(SOS2,G2,signal);
figure;
subplot(311);plot(t(1:100),signal(1:100));grid on;
title('Original signal');xlabel('Time (s)');ylabel('Amplitude');

subplot(312);plot(t(1:100),filtered_FIR(1:100));grid on;
title('Filtered signal (FIR)');xlabel('Time (s)');ylabel('Amplitude');

subplot(313);plot(t(1:100),filtered_IIR(1:100));grid on;
title('Filtered signal (IIR)');xlabel('Time (s)');ylabel('Amplitude');

sgtitle('Comparison of FIR and IIR filtering');

%Compute FFT
N=length(signal);
f=linspace(0,fs/2,N/2);
X_original=abs(fft(signal));
X_FIR=abs(fft(filtered_FIR));
X_IIR=abs(fft(filtered_IIR));

%Normalize and keep only positive frequencies
X_original=X_original(1:N/2)/max(X_original);
X_FIR=X_FIR(1:N/2)/max(X_FIR);
X_IIR=X_IIR(1:N/2)/max(X_IIR);

figure;
subplot(311);plot(f,X_original);grid on;
title('Frequency response of original signal');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(312);plot(f,X_FIR);grid on;
title('Frequency response of filtered signal (FIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

subplot(313);plot(f,X_IIR);grid on;
title('Frequency response of filtered signal (IIR)');xlabel('Frequency (Hz)');ylabel('Magnitude');

sgtitle('Comparison of frequency responses');


% Lab 9
%% Task 1 
a=audioinfo('omni_drum.mp3');
disp(a);
whos a;

a.Title='My Song';
a.Rating=[10, 5];
mystructure=a;
mystructure.b=5;

sampleRate=a.TotalSamples/a.Duration;
disp(['Calculated Sample Rate:', num2str(sampleRate)]);

%% Task 2 - creating sinewave
fs=3000;
t=0:1/fs:2;
f=1000;
A=.5;
w=15*pi/180;
y=A*sin(2*pi*f*t+w);
sound(y,fs,16);
plot(t,y);xlabel('Time (s)');ylabel('Amplitude');
title('Sinewave with a frequency of 1000 Hz, amplitude of 0.2 and phase of 15 degree');

%% Task 3 - Phase cancellation
fs=44100;
f1=440;
f2=440;
A1=.3;
A2=.3;
t=0:1/fs:5;
w1=0*pi/180;
w2=180*pi/180;
y1=A1*sin(2*pi*f1*t+w1);
y2=A2*sin(2*pi*f2*t+w2);
y=(y1+y2)/2;
sound(y,fs,16);
plot(t,y);title('Phase cancellation');xlabel('Time (s)');ylabel('Amplitude');

% Change one frequency to 880 Hz
f2=880;
y2=A2*sin(2*pi*f2*t+w2);
y=(y1+y2)/2;
sound(y,fs,16);
figure;
subplot(221);plot(t,y);
title('Changed frequency to 880 Hz');xlabel('Time (s)');ylabel('Amplitude');

% Change one frequency to 441 Hz
f2=441;
y2=A2*sin(2*pi*f2*t+w2);
y=(y1+y2)/2;
sound(y,fs,16);
subplot(222);plot(t,y);
title('Changed frequency to 441 Hz');xlabel('Time (s)');ylabel('Amplitude');

% Phase changed to 179 degrees
w2=179*pi/180;
y2=A2*sin(2*pi*f2*t+w2);
sound(y,fs,16);
subplot(223);plot(t,y);
title('Phase changed to 179 degrees');xlabel('Time(s)');ylabel('Amplitude');

% Phase changed to 181 degrees
w2=181*pi/180;
y2=A2*sin(2*pi*f2*t+w2);
sound(y,fs,16);
subplot(224);plot(t,y);
title('Phase changed to 181 degrees');xlabel('Time(s)');ylabel('Amplitude');

%% Task 4 - Binaural beats/Cocktail party effect
fs=44100;
t=0:1/fs:5;
f1=300;
f2=310;
w=0*pi/180;
A=.5;
y1=A*sin(2*pi*f1*t+w);
y2=A*sin(2*pi*f2*t+w);
y=[y1;y2];
sound(y,fs,16);

% Change frequency difference to 2 Hz
f2=f1+2;
y2=A*sin(2*pi*f2*t+w);
y=[y1;y2];
sound(y,fs,16);

% Play 1000 Hz and 1010 Hz
f1=1000;
f2=1010;
y1=A*sin(2*pi*f1*t+w);
y2=A*sin(2*pi*f2*t+w);
y=[y1;y2];
sound(y,fs,16);

% Create cocktail party effect
f1=300;
f2=330;
y1=A*sin(2*pi*f1*t+w);
y2=A*sin(2*pi*f2*t+w);
y=[y1;y2];
sound(y,fs,16);

%% Task 5 - Complex tones via additive synthesis
fs=44100;
t=0:1/fs:5;
f1=400; f2=2*f1; f3=3*f1; f4=4*f1;
A1=.3; A2=A1/2; A3=A1/3; A4=A1/4;
w=0;
y1=A1*sin(2*pi*f1*t+w);
y2=A2*sin(2*pi*f2*t+w);
y3=A3*sin(2*pi*f3*t+w);
y4=A4*sin(2*pi*f4*t+w);
y=(y1+y2+y3+y4)/4;
sound(y,fs,16);

%% Task 6 - Melodies
fs=44100;
t=0:1/fs:5;
A1=.3; A2=A1/2; A3=A1/3; A4=A1/4;
f1=440; f2=f1*2; f3=f1*3; f4=f1*4;
w=0;
y1=A1*sin(2*pi*f1*t+w);
y2=A2*sin(2*pi*f2*t+w);
y3=A3*sin(2*pi*f3*t+w);
y4=A4*sin(2*pi*f4*t+w);
y=[y1 y2 y3 y4];
soundsc(y,fs);

x=y(1:2:end);
soundsc(x,fs);

%% Task 6 - Melodies 2nd way
fs=44100;
t=0:1/fs:5;
notes=[440, 494, 523, 587];
melody=[];
for f=notes
    melody=[melody, sin(2*pi*f*t)];
end
soundsc(melody,fs);
x=melody(1:2:end);
soundsc(x,fs);

%% Task 7 - Modulation
fs=44100;
t=0:1/fs:5;
f=440;
y=sin(2*pi*f*t);
% frequency modulation
fm=modulate(y,20,fs,'fm');
soundsc(fm,fs);
% amplitude modulation
am=modulate(y,.5,fs,'am');
soundsc(am,fs);
% phase modulation
pm=modulate(y,20,fs,'pm');
soundsc(pm,fs);

%% Task 8 - FFT
fs=44100;
t=0:1/fs:5;
f=440;
y=sin(2*pi*f*t);
N=fs;
Y=fft(y,N)/N;
magTransform=abs(Y);
faxis=linspace(-fs/2,fs/2,N);
plot(faxis, fftshift(magTransform));
xlabel('Frequency (Hz)');
axis([0 2000 0 max(magTransform)]);

%% Task 9 - Spectrogram
fs=44100;
t=0:1/fs:5;
f=440;
y=sin(2*pi*f*t);
win=128;
hop=win/2;
nfft=win;
spectrogram(y,win,hop,nfft,fs,'yaxis');

% quadratic chirp
c=chirp(t,0,1,440,'q');
spectrogram(c,win,hop,nfft,fs,'yaxis');

%% Task 10 - Shephard Tones
fs=16000;
d=1;
t=0:1/fs:d-1/fs;
fmin=300;
fmax=3400;
n=12;
l=mod(([0:n-1]/n)'*ones(1,fs*d)+ones(n,1)*(t/(d*n)),1);
f=fmin*(fmax/fmin).^l;
p=2*pi*cumsum(f,2)/fs;
p=diag((2*pi*floor(p(:,end)/(2*pi)))./p(:,end))*p;
s=sin(p);
a=0.5-0.5*cos(2*pi*l);
w=sum(s.*a)/n;
w=repmat(w,1,3);
specgram(w,2048,fs,2048,1800);
ylim([0 4000]);
soundsc(w,fs);
audiowrite('shephard.wav',w,fs);

%% Task 11 - Amplitude modulation and Ring modulation
load handel;
index=1:length(y);
Fc=5;
A=.5;
w=0;
trem=(w*pi/180+A*sin(2*pi*index*(Fc/Fs)))';
y=y.*trem;
soundsc(y,Fs);

Fc=1000;
trem=(w*pi/180+A*sin(2*pi*index*(Fc/Fs)))';
y=y.*trem;
soundsc(y,Fs);

%% Task 12 - Filter design
%% Task 12 - Filter design
fs=44100;
t=0:1/fs:5;
y=sin(2*pi*500*t)+sin(2*pi*2500*t);
noise=0.5*randn(size(t));
y=y+noise;
% Lowpass filter cutoff 442 order 10
cutoff=442/(fs/2);
order=10;
d=designfilt('lowpassfir','CutoffFrequency',cutoff,'FilterOrder',order);
o=filter(d,y);
soundsc(o,fs);

figure;
subplot(221);plot(y(1:800));xlabel('Sample Index');ylabel('Amplitude');
title('Original signal (first 800 samples)');

subplot(222);plot(o(1:800));xlabel('Sample Index');ylabel('Amplitude');
title('Filtered signal (Lowpass, cutoff 442 Hz, order 10)');

% Change cutoff and filter order
cutoff=600/(fs/2);
order=20;
d=designfilt('lowpassfir','CutoffFrequency',cutoff,'FilterOrder',order);
o2=filter(d,y);
soundsc(o2,fs);

subplot(223);plot(o2(1:800));xlabel('Sample Index');ylabel('Amplitude');
title('Filtered signal (Lowpass, cutoff 600 Hz, order 20)');

% Highpass filter cutoff 420 order 10
cutoff=442/(fs/2);
order=10;
d_high=designfilt('highpassfir','CutoffFrequency',cutoff,'FilterOrder',order);
o_high=filter(d_high,y);
soundsc(o_high,fs);

subplot(224);plot(o_high(1:800));xlabel('Sample Index');ylabel('Amplitude');
title('Filtered signal (Highpass, cutoff 442 Hz, order 10)');


Companding
Companding is a technique for reducing audio data rates by using unequal quantization levels. Since the human ear can detect sounds from 0 dB SPL to 120 dB SPL (a million-fold amplitude range) but can only distinguish about 120 different loudness levels (spaced logarithmically), companding takes advantage of this perceptual characteristic.
The benefits are significant:
•	Telephone-quality speech requires 12 bits with equal spacing
•	Only 8 bits are required with unequal spacing that matches human hearing
•	This provides a 33% reduction in data rate
Companding can be implemented in three ways:
1.	Using a nonlinear analog circuit before a linear ADC
2.	Using an ADC with internally unequal steps
3.	Using a linear ADC followed by a digital lookup table
Two standard algorithms are used globally:
•	μ-law (mu-law): Used in North America and Japan
•	A-law: Used in Europe
Both use logarithmic nonlinearity with slight differences in their implementation.
Speech Synthesis and Recognition
Speech Production Model
Nearly all speech synthesis and recognition techniques are based on a model of human speech production that classifies sounds as:
•	Voiced sounds (like vowels): Produced when air passes through vibrating vocal cords, creating periodic puffs represented by a pulse train generator
•	Fricative sounds (like s, f, sh): Produced by air turbulence when airflow is constricted, represented by a noise generator
Both sound sources are modified by acoustic cavities (tongue, lips, mouth, throat), which can be modeled as a linear filter with resonances called format frequencies.
Spectrogram
Speech can be visualized using spectrograms (voiceprints) that show how frequency content changes over time:
•	Voiced sounds appear as regular harmonic patterns
•	Fricatives appear as noisy, random patterns
Speech Synthesis
Speech can be synthesized by specifying three parameters every 25 milliseconds:
1.	Type of excitation (periodic or random noise)
2.	Frequency of the periodic wave (if used)
3.	Coefficients of a digital filter mimicking vocal tract response
This approach enables low data rate speech synthesis (few kbits/sec) and is the basis for Linear Predictive Coding (LPC) compression, though quality can sound mechanical.
Speech Recognition Challenges
Speech recognition algorithms attempt to match patterns in extracted parameters with stored templates. Major limitations include:
1.	Requiring pauses between words (eliminating natural speech flow)
2.	Limited vocabulary (increasing vocabulary increases error rates)
3.	Speaker-dependent training requirements
Unlike humans, most algorithms recognize words based only on sound, not context. This creates significant disadvantages compared to human listeners, who use context and expectations to disambiguate similar-sounding phrases (like "spider ring" vs. "spy during").
Despite these challenges, speech recognition remains an active area of DSP research with potential to replace various input methods like typing and keyboard entry.


