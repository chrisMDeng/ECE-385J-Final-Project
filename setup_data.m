function [match_yes, match_no, match_idk, nomatch_yes, nomatch_no, nomatch_idk] = setup_data(subject)

    sub1_pre = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S001N2pre\eeg';
    sub1_post = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S001N2post\eeg';
    sub1_2 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S002N2online\eeg';
    sub1_3 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S003N2online\eeg';
    
    sub2_pre = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S001N2pre\eeg';
    sub2_post = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S001N2post\eeg';
    sub2_2 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S002N2online\eeg';
    % Issues with this %sub2_3 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S003online\eeg';
    
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
    match_idk = {};
    nomatch_yes = {};
    nomatch_no = {};
    nomatch_idk = {};
    
    if subject == 1
        used_folder = folders1;
    elseif subject == 2
        used_folder = folders2;
    elseif subject == 3
        used_folder = folders3;
    else
        disp("ERROR: Choose Subjects 1-3")
    end

    for i = 1:numel(used_folder)
        %disp(i)
        [a, b, c, d, e, f] = load_data(used_folder{i});
        match_yes = [match_yes, a];
        match_no = [match_no, b];
        match_idk = [match_idk, c];
        nomatch_yes = [nomatch_yes, d];
        nomatch_no = [nomatch_no, e];
        nomatch_idk = [nomatch_idk, f];
    end


end