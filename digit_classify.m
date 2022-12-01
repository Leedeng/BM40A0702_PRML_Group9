function pred = digit_classify(testdata,net,classes)


img_name = '2D.png';

% convert 3D-digtal into 2D-digtal
pos = testdata.pos;
fig = figure();
plot(pos(:,1),pos(:,2),'k')
axis off
h = getframe;
out = h.cdata;
out = imresize(out,[256,256]);
out = imbinarize(out,"global");
out = out * 256;
imwrite(out,img_name)
close(fig)

% import 2D-digtal
img = imread(img_name);
% normalize image
img = double(img/255);
% predict
pred = classify(net,img(:,:,1));
pred = str2double(classes(pred));
delete(img_name)