I = imread('AT3_1m4_01.tif');
figure;
imshow(I)

I = im2double(I);
I = log(1 + I);

M = 2*size(I,1) + 1;
N = 2*size(I,2) + 1;

sigma = 10;

[X, Y] = meshgrid(1:N,1:M);
centerX = ceil(N/2);
centerY = ceil(M/2);
gaussianNumerator = (X - centerX).^2 + (Y - centerY).^2;
H = exp(-gaussianNumerator./(2*sigma.^2));
H = 1 - H;

figure;
imshow(H,'InitialMagnification',25)

H = fftshift(H);

If = fft2(I, M, N);
Iout = real(ifft2(H.*If));
Iout = Iout(1:size(I,1),1:size(I,2));

Ihmf = exp(Iout) - 1;

figure;
imshowpair(I, Ihmf, 'montage')

