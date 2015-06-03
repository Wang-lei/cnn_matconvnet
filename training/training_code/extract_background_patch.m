function patch=extract_background_patch(im,box,varargin)
% return background patches from image im which are sufficiently far from
% the input box
% box: [x y width height]


% default params
param.overlap_th = 0.2 ; % overlaping with given box threashod to be considered as background
param.patch_sizew = 80 ; % return patch width
param.patch_sizeh = 80 ; % return patch height

param = vl_argparse(param, varargin);

[imh,imw,nbch]=size(im);
assert(nbch==3);

proposal_bounds= [imw-param.patch_sizew+1 imh-param.patch_sizeh+1] ;    % i and j proposal max bounds:
%                                                                       %(i,j) is the top-left corner position of the proposal

cpt=0;
while(true)
    cpt=cpt+1;
    % generate a top-left corner position
    prop_left_c=ceil(rand(1,2).*(proposal_bounds));
    prop_box= [prop_left_c param.patch_sizew param.patch_sizeh];
    overlapRatio = bboxOverlapRatio(prop_box,box,'ratioType','Min');
    if overlapRatio<param.overlap_th
        break
    end
    if cpt > 100
        error('can not find non-overlaping patch');
    end
end
patch=im(prop_box(2):prop_box(2)+prop_box(4)-1,prop_box(1):prop_box(1)+prop_box(3)-1,:);


% showimage(im);figure
% showimage(patch);figure
% RGB = insertShape(im,'FilledRectangle',prop_box,'Color','green');
% RGB = insertShape(RGB,'FilledRectangle',box,'Color','yellow');
% showimage(RGB)


