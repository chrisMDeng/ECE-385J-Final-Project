function [match_yes, match_no, nomatch_yes, nomatch_no] = setup_data(subject)

    sub1_pre = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S001N2pre\eeg';
    sub1_post = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S001N2post\eeg';
    sub1_2 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S002N2online\eeg';
    sub1_3 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S003N2online\eeg';
    
    sub2_pre = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S001N2pre\eeg';
    sub2_post = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S001N2post\eeg';
    sub2_2 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S002N2online\eeg';
    % issues with this %sub2_3 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S003online\eeg';
    
    sub3_pre = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P403\ses-S001N2pre\eeg';
    sub3_post = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P403\ses-S001N2post\eeg';
    sub3_2 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P403\ses-S002online\eeg';
    sub3_3 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P403\ses-S003N2online\eeg';
    
    folders1 = {sub1_pre, sub1_post, sub1_2, sub1_3};
    folders2 = {sub2_pre, sub2_post, sub2_2};
    folders3 = {sub3_pre, sub3_post, sub3_2, sub3_3};
    
    % Make arrays
    match_yes = {};
    match_no = {};
    nomatch_yes = {};
    nomatch_no = {};
    
    if subject == 1
        for i = 1:numel(folders1)
            [a, b, c, d] = load_data(folders1{i});
            match_yes = [match_yes, a];
            match_no = [match_no, b];
            nomatch_yes = [nomatch_yes, c];
            nomatch_no = [nomatch_no, d];
        end
    elseif subject == 2
        for i = 1:numel(folders2)
            [a, b, c, d] = load_data(folders2{i});
            match_yes = [match_yes, a];
            match_no = [match_no, b];
            nomatch_yes = [nomatch_yes, c];
            nomatch_no = [nomatch_no, d];
        end
    elseif subject == 3
        for i = 1:numel(folders3)
            [a, b, c, d] = load_data(folders3{i});
            match_yes = [match_yes, a];
            match_no = [match_no, b];
            nomatch_yes = [nomatch_yes, c];
            nomatch_no = [nomatch_no, d];
        end
    else
        disp("ERROR: Choose Subjects 1-3")
    end

end