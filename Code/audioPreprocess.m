% =============================================================
% audioPreprocess.m
% Noise reduction + VAD for deepfake detection pipeline.
% Called automatically by Feature_Extraction_2.m and
% Deepfake_Detector_Live_Main_5.m
%
% Usage:
%   y_clean = audioPreprocess(y_raw, fs);
% =============================================================

function y_out = audioPreprocess(y_in, fs)
    Fs_target = 16000;

    % 1. Mono + resample
    if size(y_in,2) > 1, y_in = mean(y_in,2); end
    if fs ~= Fs_target,  y_in = resample(y_in, Fs_target, fs); end
    y = double(y_in(:));   % force column vector

    % 2. Normalise
    peak = max(abs(y));
    if peak > eps, y = y / peak; end

    % 3. Spectral subtraction
    y = spectralSubtraction(y, Fs_target);

    % 4. VAD
    y = applyVAD(y, Fs_target);

    % 5. Re-normalise
    peak = max(abs(y));
    if peak > eps, y = y / peak; end

    y_out = y;
end

% =============================================================
% Spectral Subtraction — fully explicit indexing, no buffer()
% =============================================================
function y_out = spectralSubtraction(y, Fs)
    frameLen  = round(0.025 * Fs);   % 400 at 16kHz
    hopLen    = round(0.010 * Fs);   % 160 at 16kHz
    nFFT      = 2^nextpow2(frameLen);% 512
    halfLen   = nFFT/2 + 1;          % 257  (one-sided spectrum)
    overSub   = 1.5;
    floorCoef = 0.005;

    % Need at least one full frame
    if length(y) < frameLen
        y_out = y;
        return;
    end

    win = hann(frameLen, 'periodic');  % length = frameLen = 400

    % Noise PSD from first frame
    noiseFrame = y(1:frameLen) .* win;             % 400 samples
    noisePSD   = abs(fft(noiseFrame, nFFT)).^2;    % 512 samples
    noisePSD   = noisePSD(1:halfLen);              % 257 samples

    % Number of complete frames
    nFrames = floor((length(y) - frameLen) / hopLen) + 1;

    y_out  = zeros(length(y), 1);
    winSum = zeros(length(y), 1);

    for k = 1:nFrames
        s = (k-1)*hopLen + 1;         % start index
        e = s + frameLen - 1;         % end index

        if e > length(y), break; end  % skip incomplete last frame

        frame = y(s:e) .* win;        % frameLen = 400 samples exactly

        % Forward FFT
        X     = fft(frame, nFFT);     % 512 samples
        Xhalf = X(1:halfLen);         % 257 samples (one-sided)
        mag   = abs(Xhalf);           % 257 magnitudes
        ph    = angle(Xhalf);         % 257 phases

        % Spectral subtraction
        mag_clean = sqrt(max(mag.^2 - overSub*noisePSD, floorCoef*noisePSD));

        % Reconstruct full symmetric spectrum (must be exactly nFFT = 512)
        % [X(1)...X(N/2+1)] has 257 elements
        % mirror: [X(N/2)...X(2)] = flipud of elements 2..N/2 = 255 elements
        % total = 257 + 255 = 512 = nFFT  ✓
        Xclean_half = mag_clean .* exp(1j*ph);             % 257
        Xclean_full = [Xclean_half; ...
                       conj(flipud(Xclean_half(2:end-1)))];% 257+255 = 512

        % Inverse FFT and take real part
        frame_out = real(ifft(Xclean_full, nFFT));         % 512 samples
        frame_out = frame_out(1:frameLen) .* win;          % keep 400, re-window

        % Overlap-add
        y_out(s:e)  = y_out(s:e)  + frame_out;
        winSum(s:e) = winSum(s:e) + win.^2;
    end

    % Divide by window sum where nonzero
    idx           = winSum > 1e-8;
    y_out(idx)    = y_out(idx) ./ winSum(idx);
    y_out         = y_out(1:length(y));
end

% =============================================================
% Voice Activity Detection — explicit frame loop, no buffer()
% =============================================================
function y_out = applyVAD(y, Fs)
    frameLen  = round(0.025 * Fs);
    hopLen    = round(0.010 * Fs);
    minSpeech = round(0.25  * Fs);

    if length(y) < frameLen * 2
        y_out = y;
        return;
    end

    nFrames = floor((length(y) - frameLen) / hopLen) + 1;
    energy  = zeros(1, nFrames);
    zcr     = zeros(1, nFrames);

    for k = 1:nFrames
        s = (k-1)*hopLen + 1;
        e = s + frameLen - 1;
        if e > length(y), break; end
        frame     = y(s:e);
        energy(k) = sum(frame.^2) / frameLen;
        zcr(k)    = sum(abs(diff(sign(frame)))) / (2*frameLen);
    end

    eThresh    = prctile(energy, 15) * 5;
    zThresh    = prctile(zcr,    85) * 0.8;
    speechMask = (energy > eThresh) & (zcr < zThresh);
    speechMask = medfilt1(double(speechMask), 5) > 0.5;

    y_out = [];
    for k = 1:nFrames
        if speechMask(k)
            s = (k-1)*hopLen + 1;
            e = min(s+frameLen-1, length(y));
            y_out = [y_out; y(s:e)];
        end
    end

    if length(y_out) < minSpeech
        y_out = y;
    end
end
