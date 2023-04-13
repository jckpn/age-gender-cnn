load('wiki_crop/wiki.mat');

[ages,~]=datevec(datenum(wiki.photo_taken,7,1)-wiki.dob); % From IMDB-WIKI docs

for i = 1:length(wiki.dob)
    if isnan(wiki.gender(i)) % Skip files where gender is NaN
        continue
    end
    if wiki.gender(i) == 0 % Map gender from 0 -> F, 1 -> M
        gender = 'F';
    else
        gender = 'M';
    end
    age = num2str(ages(i));
    id = num2str(i); % Get the file id
    new_path = ['wiki_crop/' gender '_' age '_' id '.jpg']; % Create the new file name↪→
    old_path = ['wiki_crop/' char(wiki.full_path(i))]; % Get the old file name
    disp([id '/' num2str(length(wiki.dob)) ': ' old_path ' -> ' new_path]);
    try
        copyfile(old_path, new_path); % Rename the file
        delete(old_path)
    catch exception
        warning(['Error copying ' old_path ': ' exception.message]);
    end
end