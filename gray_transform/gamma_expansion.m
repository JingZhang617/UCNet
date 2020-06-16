function output_channel = gamma_expansion(input_channel)
input_channel = im2double(input_channel);
[h,w] = size(input_channel);
output_channel = zeros(h,w);
for i = 1:h
    for j = 1:w
        if(input_channel(i,j)>0.04045)
            output_channel(i,j) = ((input_channel(i,j)+0.055)/1.055)^2.4;
        else
            output_channel(i,j) = input_channel(i,j)/12.96;
        end 
    end
end
end

