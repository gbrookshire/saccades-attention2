%% Load the data

load visual_stimuli.mat


%% Check the number of stimuli in each category

for human = 0:1
    for face = 0:1
        for animal = 0:1
            for natural = 0:1
                x = ([visual_stimuli.human] == human) & ...
                    ([visual_stimuli.face] == face) & ...
                    ([visual_stimuli.animal] == animal) & ...
                    ([visual_stimuli.natural] == natural);
                x = sum(x);
                if x == 0
                    continue
                end
                fprintf('Human %i\tFace %i\tAnimal %i\tNatural %i\t%i\n',
                        human, face, animal, natural, sum(x));
            end
        end
    end
end


%% Save the stimuli as separate files

for i_img = 1:length(visual_stimuli)
    fname = sprintf('%i.jpg', i_img);
    imwrite(visual_stimuli(i_img).pixel_values, fname)
end

