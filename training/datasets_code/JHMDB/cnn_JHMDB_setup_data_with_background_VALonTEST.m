function imdb = cnn_JHMDB_setup_data_with_background_VALonTEST(opts)
splitslist = dir(sprintf('%s/*train%d.txt',opts.splitpath,opts.split));
nb_classes=length(splitslist);
assert(nb_classes==length(opts.actionsnames)) ;

iwidth =320 ;
iheight = 240;


tmpdir=dir(opts.dataDir);
num_images=length(tmpdir); clear tmpdir ;
fprintf('Initialize with %d files\n',num_images);

%imdb.images.data = single(-ones(net.normalization.imageSize(1), net.normalization.imageSize(2), net.normalization.imageSize(3), num_images));
imdb=[] ;
imdb.images.label = single(zeros(num_images,1));
imdb.images.name = cell(num_images,1) ;
imdb.images.name(:) = {[opts.dataDir '/']} ;
imdb.images.bbox = cell(num_images,1) ;

added=0;
for c = 1 :nb_classes
    fprintf('class %d out of %d\n',c,length(splitslist));
    
    vid_id = 0 ;
    
    split_id = fopen(sprintf('%s/%s',opts.splitpath,splitslist(c).name)) ;
    
    [sample,label] = strtok(fgetl(split_id)); label = str2double(label) ;
    while ischar(sample)
        if label == 1
            cn_im_ = (dir(sprintf('%s/%s_im0*.jpg',opts.dataDir,sample))) ;
            imdb.images.name(added+1:added+length(cn_im_))=strcat(imdb.images.name(added+1:added+length(cn_im_)), {cn_im_.name}') ;
            imdb.images.label(added+1:added+length(cn_im_))=c ;
            
            % get joints
            joints = load(sprintf('%s/%s/%s/joint_positions',opts.joitsDir,opts.actionsnames{c},sample));
            for ii=1:length(cn_im_)
                [~,cna,~]=fileparts(cn_im_(ii).name) ;
                imagepositions=str2num(cna(strfind(cna,'_im0')+4:end));
                if imagepositions > size(joints.pos_img,3)
                    break;
                end
                [lhand,rhand,upbody,fullbody]=get_boxes(joints.pos_img(:,:,imagepositions),joints.scale(imagepositions),iwidth,iheight);
                fullimagebox = [0 0 iwidth iheight];
                ov=bboxOverlapRatio(fullbody,fullimagebox);
                if ov > 0.3
                    continue; % do not add if fullbody takes more than 30% of the frame
                end
                imdb.images.bbox{added+ii} =[lhand;rhand;upbody;fullbody];
                
                %imm=imread(sprintf('%s/%s',opts.originalProposalDataDir,cn_im_(ii).name));RGB = insertShape(imm,'FilledRectangle',fullbody,'Color','green');RGB = insertShape(RGB,'FilledRectangle',upbody,'Color','yellow');RGB = insertShape(RGB,'FilledRectangle',rhand,'Color','red');RGB = insertShape(RGB,'FilledRectangle',lhand,'Color','blue');
                %showimage(RGB);
                %hold on;
                %scatter(joints.pos_img(1,:,imagepositions),joints.pos_img(2,:,imagepositions))
                %scatter(joints.pos_img(1,12,imagepositions),joints.pos_img(2,12,imagepositions),100,[1 0 0])
                %scatter(joints.pos_img(1,13,imagepositions),joints.pos_img(2,13,imagepositions),100,[0 0 1])
                
            end
            
            
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
    imdb.images.bbox=imdb.images.bbox(1:added);
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
imdb.images.bbox(end+1:end+nbimtest) = cell(nbimtest,1) ;

% add background images
nb_background_image = opts.backgroundnum ;
nb_valid_bg_image=sum(~cellfun(@isempty,imdb.images.bbox) .* (imdb.images.set == 1)');
background_per_train_image=round(nb_background_image/nb_valid_bg_image);
numbackimage=length(imdb.images.name)*background_per_train_image;
backimages=[];
backimages.label= single((nb_classes+1)*ones(numbackimage,1)); % background class
backimages.name= cell(numbackimage,1) ;
backimages.name(:)=  {[opts.originalProposalDataDir '/']} ;
backimages.bbox= cell(numbackimage,1) ;
backimages.set= ones(numbackimage,1) ;

for i=1:length(imdb.images.name)
    if imdb.images.set(i) ~= 1 || isempty(imdb.images.bbox{i})
        continue;
    end
    
    % duplicate the images (random patches will be croped later)
    indc= (i-1)*background_per_train_image+1:(i-1)*background_per_train_image+background_per_train_image;
    [~,cimn,~]=fileparts(imdb.images.name{i}) ;
    backimages.name(indc) = strcat(backimages.name(indc),[cimn '.jpg']) ;
    backimages.bbox(indc) = imdb.images.bbox(i) ;
    
    % remove joint from original class
    imdb.images.bbox{i} = [];
    
end

noempty = ~cellfun(@isempty,backimages.bbox);

backkeepnum=sum(noempty);
imdb.images.label(end+1:end+backkeepnum)= backimages.label(noempty);
imdb.images.name(end+1:end+backkeepnum)= backimages.name(noempty);
imdb.images.bbox(end+1:end+backkeepnum)= backimages.bbox(noempty);
imdb.images.set(end+1:end+backkeepnum)= backimages.set(noempty);




function [lhand,rhand,upbody,fullbody]=get_boxes(positions,sc,iwidth,iheight)
lhandposition=13;
rhandposition=12;
upbodypositions=[1 2 3 4 5 6 7 8 9 12 13];
lsideCNNf=40;

lside = lsideCNNf*sc;
lhand = [positions(:,lhandposition)'-lside,2*lside,2*lside] ;
rhand = [positions(:,rhandposition)'-lside,2*lside,2*lside] ;
lhand=max(lhand,0);lhand(1)=min(lhand(1),iwidth);lhand(2)=min(lhand(2),iheight);
rhand=max(rhand,0);rhand(1)=min(rhand(1),iwidth);rhand(2)=min(rhand(2),iheight);


lside=3/4*lsideCNNf*sc ;
mi=min(positions(:,upbodypositions),[],2)'-lside;
ma=max(positions(:,upbodypositions),[],2)'+lside;
mi=max(mi,0);mi(1)=min(mi(1),iwidth);mi(2)=min(mi(2),iheight);
ma=max(ma,0);ma(1)=min(ma(1),iwidth);ma(2)=min(ma(2),iheight);
upbody =   [mi,ma-mi];

mi=min(positions,[],2)'-lside;
ma=max(positions,[],2)'+lside;
mi=max(mi,0);mi(1)=min(mi(1),iwidth);mi(2)=min(mi(2),iheight);
ma=max(ma,0);ma(1)=min(ma(1),iwidth);ma(2)=min(ma(2),iheight);
fullbody = [mi,ma-mi];
