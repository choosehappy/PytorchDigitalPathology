%%
files=dir('*.png');

nfiles=length(files);
for fi=1:nfiles
    mask=imread(files(fi).name);
    io=imread(sprintf('../imgs/%s',strrep(files(fi).name,'_class.png','.tif')));
    io=imresize(io,[size(mask,1),size(mask,2)]);
    fused=imfuse(mask,io);
    imwrite(fused,strrep(files(fi).name,'_class.png','_fused.png'));
end