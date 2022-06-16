# Text2EL

This repository contains the source code of the Text2EL prototype developed for our paper titled "Exploiting Unstructured Text for Event Log Enrichment and Enhancement". Text2EL is an event log enrichment approach using unstructured text. Please refer to the paper for more details.

# Installation 
Install the general and specialized python libraries mentioned in requirements.txt

# Dataset description
Details about the dataset are avaiable inside the data_description file.

# Phase 1: Events and attributes extraction 
extract_events_attributes.py extracts the events and attributes from extracted note collection.

extract_note_collections.py filters and extracts the required notes from the MIMIC-III dataset. 

# Phase 2 : Event verification and enrichment
verify_events.py verifies the similarity, adds new events and correct timestamps. Coded based on MIMIC-III dataset.

