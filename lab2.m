%% lab2
clc          
clear all    
close all    

% Turimi duomenys
x = 0.1:1/22:1;   % Sukuria įėjimo duomenų vektorių nuo 0.1 iki 1 su 22 intervalais
y = ((1+0.6*sin(2*pi*x/0.7))+0.3*sin(2*pi*x))/2;   % Norima išėjimo reikšmė pagal pateiktą formulę
figure(1)        % Sukuria naują paveikslą
plot(x,y)        % Braižo x ir y duomenų grafiką
grid on          % Įjungia tinklelį grafike

% Svoriai
% Pirmas sluoksnis (paslėptasis sluoksnis)
w11_1 = rand(1); b1_1 = rand(1); % Atsitiktiniai svoriai ir poslinkis (bais) neuronui
w21_1 = rand(1); b2_1 = rand(1); % Antras neuronas
w31_1 = rand(1); b3_1 = rand(1); % Trečias neuronas
w41_1 = rand(1); b4_1 = rand(1); % Ketvirtas neuronas
w51_1 = rand(1); b5_1 = rand(1); % Penktas neuronas

% Antras sluoksnis (išėjimo sluoksnis)
w11_2 = rand(1); 
w12_2 = rand(1);
w13_2 = rand(1);
w14_2 = rand(1);
w15_2 = rand(1);
b1_2 = rand(1); % Išėjimo sluoksnio poslinkis

eta = 0.1;      % Mokymosi koeficientas
Y = zeros(1, length(x)); % Pradinis išėjimo vektorius

% Atsakas
for j = 1:2000   % Mokymosi ciklų skaičius
    for i = 1:length(x) % Iteruoja per kiekvieną įėjimo reikšmę
        % Atsako skaičiavimas paslėptajame sluoksnyje
        v1_1 = w11_1 * x(i) + b1_1; % Pirmo paslėptojo neurono grynasis įėjimas
        v2_1 = w21_1 * x(i) + b2_1; % Antro neurono grynasis įėjimas
        v3_1 = w31_1 * x(i) + b3_1;
        v4_1 = w41_1 * x(i) + b4_1;
        v5_1 = w51_1 * x(i) + b5_1;

        % Aktyvavimo funkcija paslėptajame sluoksnyje
        y1_1 = tanh(v1_1); % Hiperbolinio tangento aktyvavimo funkcija pirmam neurone
        y2_1 = tanh(v2_1); 
        y3_1 = tanh(v3_1); 
        y4_1 = tanh(v4_1); 
        y5_1 = tanh(v5_1);

        % Antras sluoksnis
        v1_2 = y1_1 * w11_2 + y2_1 * w12_2 + y3_1 * w13_2 + y4_1 * w14_2 + y5_1 * w15_2 + b1_2; % Išėjimo neurono grynasis įėjimas
        y1_2 = v1_2; % Tiesinė aktyvavimo funkcija išėjimo neurone
        Y(i) = y1_2; % Įrašome rezultatą į išėjimo vektorių

        % Klaidos skaičiavimas
        e = y(i) - y1_2; % Apskaičiuojame klaidą tarp norimos ir esamos išėjimo reikšmės

        % Svorių atnaujinimas išėjimo sluoksnyje
        delta1_2 = e; % Klaida išėjimo sluoksnyje
        
        % Paslėptojo sluoksnio klaidos gradientas
        delta1_1 = (1 - tanh(v1_1)^2) * delta1_2 * w11_2;
        delta2_1 = (1 - tanh(v2_1)^2) * delta1_2 * w12_2;
        delta3_1 = (1 - tanh(v3_1)^2) * delta1_2 * w13_2;
        delta4_1 = (1 - tanh(v4_1)^2) * delta1_2 * w14_2;
        delta5_1 = (1 - tanh(v5_1)^2) * delta1_2 * w15_2;

        % Svorių atnaujinimas (išėjimo sluoksnis)
        w11_2 = w11_2 + eta * delta1_2 * y1_1;
        w12_2 = w12_2 + eta * delta1_2 * y2_1;
        w13_2 = w13_2 + eta * delta1_2 * y3_1;
        w14_2 = w14_2 + eta * delta1_2 * y4_1;
        w15_2 = w15_2 + eta * delta1_2 * y5_1;
        b1_2 = b1_2 + eta * delta1_2;

        % Svorių atnaujinimas (paslėptasis sluoksnis)
        w11_1 = w11_1 + eta * delta1_1 * x(i);
        w21_1 = w21_1 + eta * delta2_1 * x(i);
        w31_1 = w31_1 + eta * delta3_1 * x(i);
        w41_1 = w41_1 + eta * delta4_1 * x(i);
        w51_1 = w51_1 + eta * delta5_1 * x(i);
        b1_1 = b1_1 + eta * delta1_1;
        b2_1 = b2_1 + eta * delta2_1;
        b3_1 = b3_1 + eta * delta3_1;
        b4_1 = b4_1 + eta * delta4_1;
        b5_1 = b5_1 + eta * delta5_1;
    end
end

hold on
plot(x, Y) % Atvaizduoja mokymo rezultatus
hold off

% Testuojame neuronų tinklą su nauju įėjimo vektoriumi
x_test = 0.1:1/220:1; % Nauji testavimo duomenys
for i = 1:length(x_test)
    % Tas pats procesas kaip mokymosi etape, bet be svorių atnaujinimo
    v1_1 = w11_1 * x_test(i) + b1_1;
    v2_1 = w21_1 * x_test(i) + b2_1;
    v3_1 = w31_1 * x_test(i) + b3_1;
    v4_1 = w41_1 * x_test(i) + b4_1;
    v5_1 = w51_1 * x_test(i) + b5_1;

    y1_1 = tanh(v1_1);
    y2_1 = tanh(v2_1);
    y3_1 = tanh(v3_1);
    y4_1 = tanh(v4_1);
    y5_1 = tanh(v5_1);

    v1_2 = y1_1 * w11_2 + y2_1 * w12_2 + y3_1 * w13_2 + y4_1 * w14_2 + y5_1 * w15_2 + b1_2;
    y1_2 = v1_2;
    Y(i) = y1_2;
end

hold on
plot(x_test, Y) % Atvaizduojame tinklo atsaką naujiems duomenims
hold off
legend('Target', 'Predicted', 'Test') % Pridedame legendą
