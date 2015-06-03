function imdb = cnn_JHMDB_get_test(opts)
splitslist = dir(sprintf('%s/*test%d.txt',opts.splitpath,opts.split));
nb_classes=length(splitslist);
assert(nb_classes==length(opts.actionsnames)) ;

tmpdir=dir(opts.dataDir);
num_images=length(tmpdir); clear tmpdir ;
fprintf('Initialize with %d files\n',num_images);

imdb.images.label = single(zeros(num_images,1));
imdb.images.name = cell(num_images,1) ;
imdb.images.name(:) = {[opts.dataDir '/']} ;

added=0;
for c = 1 :nb_classes
    fprintf('class %d out of %d\n',c,length(splitslist));
    
    split_id = fopen(sprintf('%s/%s',opts.splitpath,splitslist(c).name)) ;
    
    [sample,label] = strtok(fgetl(split_id)); label = str2double(label) ;
    while ischar(sample)
        if label == 1 
            cn_im_ = (dir(sprintf('%s/%s_im*.jpg',opts.dataDir,sample))) ;
            imdb.images.name(added+1:added+length(cn_im_))=strcat(imdb.images.name(added+1:added+length(cn_im_)), {cn_im_.name}') ;
            imdb.images.label(added+1:added+length(cn_im_))=c ;
            
            added = added + length(cn_im_) ;
        end
        [sample,label] = strtok(fgetl(split_id)); label = str2double(label) ;
    end
    fclose(split_id);
end

if(added<num_images)
    imdb.images.name=imdb.images.name(1:added);
    imdb.images.label=imdb.images.label(1:added);
end

imdb.images.set = 3*ones(1, added) ;
imdb.imageDir = opts.dataDir ;
end
