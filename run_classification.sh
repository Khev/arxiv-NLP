"""  
   Colin: the below runs the scripts for the article classification. The title and abstracts should be work          ing as is. But for the fulltext you will need to make two small adjustments

	  1. Write a function that reads in the i-th fulltext as a string
             I've put this in line 59 of sentence-prediction-bert-pretrained.py. The 
             function is called on line 159, in the 'process_data_sub' function,                                        (if you're curious / need to debug)

          2. Figure out what length the fulltexts should be truncuated to. I have defined a placeholder of              500 on line 240. I did this by plotting a histogram of their lengths (in terms of tokenes) and             then picking a L* below which I have most ~= 90% of the data. However, I expect we will run in             to memory issues if we pick a large L*.... this might be undoable, actually


	  3. On line 14 of, you can change f_metadata to where you have the metatdata stored
"""

#Run test: arguments are N, data_type, where data_type = [title,abstract, fulltext]
python sentence-prediction-bert-pretrained.py 100 'title'


#Main -- full N = 10^7 graphs all the data -- uncomment when read
#python sentence-prediction-bert-pretrained.py 10000000 'title'
#python sentence-prediction-bert-pretrained.py 10000000 'abstract'
#python sentence-prediction-bert-pretrained.py 10000000 'fulltext'


