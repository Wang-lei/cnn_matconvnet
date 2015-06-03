function imdb = cnn_JHMDB_setup_data_VALonTEST(opts)
splitslist = dir(sprintf('%s/*train%d.txt',opts.splitpath,opts.split));
nb_classes=length(splitslist);
assert(nb_classes==length(opts.actionsnames)) ;

tmpdir=dir(opts.dataDir);
num_images=length(tmpdir); clear tmpdir ;
fprintf('Initialize with %d files\n',num_images);

%imdb.images.data = single(-ones(net.normalization.imageSize(1), net.normalization.imageSize(2), net.normalization.imageSize(3), num_images));
imdb=[] ;
imdb.images.label = single(zeros(num_images,1));
imdb.images.name = cell(num_images,1) ;
imdb.images.name(:) = {[opts.dataDir '/']} ;

added=0;
for c = 1 :nb_classes
    fprintf('class %d out of %d\n',c,length(splitslist));
    
    vid_id = 0 ;
    
    split_id = fopen(sprintf('%s/%s',opts.splitpath,splitslist(c).name)) ;
    
    [sample,label] = strtok(fgetl(split_id)); label = str2double(label) ;
    while ischar(sample)
        if label == 1 
            cn_im_ = (dir(sprintf('%s/%s_im*.jpg',opts.dataDir,sample))) ;
            imdb.images.name(added+1:added+length(cn_im_))=strcat(imdb.images.name(added+1:added+length(cn_im_)), {cn_im_.name}') ;
            imdb.images.label(added+1:added+length(cn_im_))=c ;
            
            added = added + length(cn_im_) ;
            vid_id=vid_id+1;
        end
        [sample,label] = strtok(fgetl(split_id)); label = str2double(label) ;
    end
    fclose(split_id);
end

if(added<num_images)
    imdb.images.name=imdb.images.name(1:added);
    imdb.images.label=imdb.images.label(1:added);
end


imdb.images.set = ones(1, added) ;
imdb.imageDir = opts.dataDir ;

% Validate on test set
imdbtest = cnn_JHMDB_get_test(opts) ;
imdbtest.images.set(:)=2 ;

nbimtest = length(imdbtest.images.set) ;
imdb.images.label(end+1:end+nbimtest) = imdbtest.images.label ;
imdb.images.name(end+1:end+nbimtest) = imdbtest.images.name ;  
imdb.images.set(end+1:end+nbimtest) = imdbtest.images.set ;  

end
