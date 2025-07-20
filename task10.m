clc;
f = 45;
Fs = 100;
duration = 2;
t = 0:1/Fs:duration;
x = sin(2*pi*f*t);
echo_delay = 0.2;
echo_amplitude = 0.5;
delay_samples = round(echo_delay*Fs);
echo = [zeros(1, delay_samples), echo_amplitude*x];
x_original = [x, zeros(1, delay_samples)];
y = x_original+echo;

sound(y, 8000);
pause(duration + echo_delay + 0.5);

% Plot original + echo in time domain
figure;
subplot(2,1,1);
plot((0:length(y)-1)/Fs, y);
title('Signal with Echo');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot in frequency domain
Y_fft = abs(fft(y));
f_axis = linspace(0, Fs, length(Y_fft));
subplot(2,1,2);
plot(f_axis, Y_fft);
title('Magnitude Spectrum of Signal with Echo');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');

% --- Complex Cepstrum Analysis ---
[xhat, nd] = cceps(y);

% Plot cepstrum
figure;
q = (0:length(xhat)-1)/Fs;
plot(q, xhat);
title('Complex Cepstrum of Signal with Echo');
xlabel('Quefrency (s)');
ylabel('Amplitude');

% Find and suppress echo peak
% (echo is at quefrency = 0.2s)
% Simple method: zero out region around echo
cepstrum_cleaned = xhat;
echo_region = round(0.2*Fs);    % Index near 0.2s
cepstrum_cleaned(echo_region-1:echo_region+1) = 0;

% Reconstruct signal from cleaned cepstrum
y_clean = icceps(cepstrum_cleaned);

% Play filtered signal
sound(real(y_clean), 8000);
pause(duration + echo_delay + 0.5);

% Plot filtered signal in time and frequency
figure;
subplot(2,1,1);
plot((0:length(y_clean)-1)/Fs, real(y_clean));
title('Signal After Echo Removal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
Y_clean_fft = abs(fft(real(y_clean)));
plot(f_axis, Y_clean_fft);
title('Magnitude Spectrum After Echo Removal');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');

%% task 1.2 
clc;
% --- Load another audio signal ---
[x2, Fs2] = audioread('singing.wav');  % load signal
x2 = x2(:,1);  % use only one channel if stereo

% Play original with echo
sound(x2, Fs2);
pause(length(x2)/Fs2 + 0.5);

% --- Plot BEFORE Echo Removal ---
figure;
subplot(2,1,1);
plot((0:length(x2)-1)/Fs2, x2);
title('Echoed Signal (Time Domain)');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
Y2_fft = abs(fft(x2));
N = length(Y2_fft);
f_axis2 = linspace(0, Fs2/2, floor(N/2));  % frequency axis up to Fs/2
plot(f_axis2, Y2_fft(1:floor(N/2)));
title('Echoed Signal (Frequency Domain)');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');

% --- Complex cepstrum analysis ---
[xhat2, nd2] = cceps(x2);

% Zero out echo region in cepstrum (around 0.3s)
cepstrum_cleaned2 = xhat2;
region2 = round(0.3 * Fs2);
cepstrum_cleaned2(region2-2:region2+2) = 0;

% Inverse complex cepstrum
y2_clean = icceps(cepstrum_cleaned2, nd2);

% Play denoised signal
sound(real(y2_clean), Fs2);
pause(length(y2_clean)/Fs2 + 0.5);

% --- Plot AFTER Echo Removal ---
figure;
subplot(2,1,1);
plot((0:length(y2_clean)-1)/Fs2, real(y2_clean));
title('Filtered Audio After Echo Removal (Time Domain)');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
Y2_clean_fft = abs(fft(real(y2_clean)));
N2 = length(Y2_clean_fft);
f_axis_clean = linspace(0, Fs2/2, floor(N2/2));
plot(f_axis_clean, Y2_clean_fft(1:floor(N2/2)));
title('Filtered Audio After Echo Removal (Frequency Domain)');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');
