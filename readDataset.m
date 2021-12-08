function [data, targets] = readDataset(imagesPaths,categories)

    ind = 1;
    for i = 1 : size(imagesPaths,2)
        for j = 1 : size(imagesPaths(i).ImageLocation,2)
            dataset{ind, 1} = imagesPaths(i).ImageLocation{j};
            if(~isempty(strfind(imagesPaths(i).Description, categories{1})))
                dataset{ind, 2} = categories{1};
                dataset{ind, 3} = 1;
            elseif(~isempty(strfind(imagesPaths(i).Description, categories{2})))
                dataset{ind, 2} = categories{2};
                dataset{ind, 3} = 2;
            elseif(~isempty(strfind(imagesPaths(i).Description, categories{4})))
                dataset{ind, 2} = categories{4};
                dataset{ind, 3} = 4;
            elseif(~isempty(strfind(imagesPaths(i).Description, categories{3})))
                dataset{ind, 2} = categories{3};
                dataset{ind, 3} = 3;
            elseif(~isempty(strfind(imagesPaths(i).Description, categories{5})))
                dataset{ind, 2} = categories{5};
                dataset{ind, 3} = 5;
            elseif(~isempty(strfind(imagesPaths(i).Description, categories{6})))
                dataset{ind, 2} = categories{6};
                dataset{ind, 3} = 6;
            end
            ind = ind + 1;
        end
    end

    data = cell(size(dataset,1),1);
    targets = zeros(length(categories), size(dataset,1));
    for i = 1 : size(dataset,1)
        data{i} = imresize(imread(dataset{i,1}),[30, 40]);
        targets(dataset{i,3},i) = 1;
    end
    
end

