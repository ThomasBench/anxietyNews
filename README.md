# anxietyNews
Semester's project at EPFL in the Digital Humanities lab, supervised by Elena Fernandez Fernande and Jérôme Baudry.


Using newspapers issued during the Second industrial revolution, this study aims to quantify
the anxiety of the people in four different languages: French, English, German, and Spanish. By
performing word ratio for OCR assessment, the quality of each newspaper was assessed and robustly
compared to the others. Then, using state-of-the-art Transformers models for date detection, it
assesses the performance of these models on noisy text datasets in French and English, and use them
to perform date detection on the French newspaper Le Figaro. From it, this study tries to tie together
the number of dates expressed in a newspaper to the anxiety of the population at that time in history.

## FILES

In this repo are :
- 1. the Creating/pre-processing notebook that was used to extract and pre-process the ocr readings of each newspapers into the results folder
- 2. the OCR assessment notebook that was used to assess the OCR quality of each pre-processed file.
- 3. The NER predictions notebook that was used to detect dates in the pre-processed files.
- 4. The Ner assessment notebook that was used to assess the NER date detections method.

All required python libraries can be read in the requirements.txt file
