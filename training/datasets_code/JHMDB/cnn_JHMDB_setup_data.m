function imdb = cnn_JHMDB_setup_data(opts)

splitslist = dir(sprintf('%s/*train%d.txt',opts.splitpath,opts.split));
nb_classes=length(splitslist);
assert(nb_classes==length(opts.actionsnames)) ;


setenv('filepath',opts.dataDir);
[~,num_images]=system('ls $filepath | wc -l');num_images=str2num(num_images);

imdb.images.label = single(zeros(num_images,1));
imdb.images.name = cell(num_images,1) ;
imdb.images.name(:) = {[opts.dataDir '/']} ;

split_videosample_ids = cell(nb_classes,1);
added=0;
for c = 1 :nb_classes
    fprintf('class %d out of %d\n',c,length(splitslist));
    
    videosample_ids = zeros(1000,2) ;
    vid_id = 0 ;
    
    split_id = fopen(sprintf('%s/%s',opts.splitpath,splitslist(c).name)) ;
    
    [sample,label] = strtok(fgetl(split_id)); label = str2double(label) ;
    while ischar(sample)
        if label == 1 
            cn_im_ = (dir(sprintf('%s/%s_im*.jpg',opts.dataDir,sample))) ;
            imdb.images.name(added+1:added+length(cn_im_))=strcat(imdb.images.name(added+1:added+length(cn_im_)), {cn_im_.name}') ;
            imdb.images.label(added+1:added+length(cn_im_))=c ;
            
            videosample_ids(vid_id+1,:) = [added+1 added+length(cn_im_)] ;
            
            added = added + length(cn_im_) ;
            vid_id=vid_id+1;
        end
        [sample,label] = strtok(fgetl(split_id)); label = str2double(label) ;
    end
    fclose(split_id);
    split_videosample_ids{c} = videosample_ids(1:vid_id,:) ;
end

if(added<num_images)
    imdb.images.name=imdb.images.name(1:added);
    imdb.images.label=imdb.images.label(1:added);
end


imdb.images.set = ones(1, added) ;


% validation samples (from the same video)
rng(0) ; 
val_p = 0.15 ;
for c = 1 :nb_classes
    split_id = split_videosample_ids{c} ;
    nb_vid_in_split=size(split_id,1) ;
    vper = randperm(nb_vid_in_split) ;
    nb_val = round(nb_vid_in_split*val_p) ;
    val = vper(1:nb_val) ;
    
    im_from_vid_idx = split_id(val,:) ;
    for ii=1:nb_val
        imdb.images.set(im_from_vid_idx(ii,1):im_from_vid_idx(ii,2)) = 2 ;
    end
end

imdb.imageDir = opts.dataDir ;
end
