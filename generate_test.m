clear;close all;
%% settings
folder = 'Test/Set5';
savepath_gt = fullfile(folder, 'gt');
savepath_2x = fullfile(folder, 'bicubic_2x');
savepath_3x = fullfile(folder, 'bicubic_3x');
savepath_4x = fullfile(folder, 'bicubic_4x');
mkdir(savepath_gt);
mkdir(savepath_2x);
mkdir(savepath_3x);
mkdir(savepath_4x);

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    image = imread(fullfile(folder,filepaths(i).name));
    file_name = filepaths(i).name;
    
    [hei,wid,n_dim] = size(image);
    image = image(1:hei-mod(hei, 12), 1:wid-mod(wid, 12), :);
    
    if size(image, 3) == 3
        image_ycbcr = rgb2ycbcr(image);
    else
        % gray-scale image
        image_ycbcr = rgb2ycbcr(cat(3, image, image, image));
    end
    
    image_ycbcr = im2double(image_ycbcr);
    
    color = image_ycbcr(:,:,2:3);
    image_y = image_ycbcr(:,:,1);
    
    [hei, wid] = size(image_y);
    
    name_gt = sprintf('%s/%s', savepath_gt, file_name);
    imwrite(image_ycbcr, name_gt);
    
    image_2x = imresize(imresize(image_y,1/2,'bicubic'),[hei,wid],'bicubic');
    name_2x = sprintf('%s/%s', savepath_2x, file_name);
    imwrite(cat(3, image_2x, color), name_2x);
    
    image_3x = imresize(imresize(image_y,1/3,'bicubic'),[hei,wid],'bicubic');
    name_3x = sprintf('%s/%s', savepath_3x, file_name);
    imwrite(cat(3, image_3x, color), name_3x);
    
    image_4x = imresize(imresize(image_y,1/4,'bicubic'),[hei,wid],'bicubic');
    name_4x = sprintf('%s/%s', savepath_4x, file_name);
    imwrite(cat(3, image_4x, color), name_4x);
end