import csv

def filter_and_update_csv(input_file, output_file, genres):
    updated_rows = []
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            last_entry = row[-1]
            if any('-' in genre for genre in genres):
                for genre in genres:
                    if '-' in genre and genre.split('-')[0].lower() in last_entry.lower():
                        last_entry = genre
            if last_entry.lower() in (genre.lower() for genre in genres):
                row[-1] = last_entry.capitalize()
                updated_rows.append(row)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(updated_rows)

# Example usage:
input_file = 'cleanedNewData.csv'
output_file = 'finalData.csv'
genres = ['Electronic', 'Pop', 'Experimental', 'Industrial', 'World', 'Latin', 'HipHop', 'Hip-Hop', 'Rap', 'ElectroHouse', 'Folk', 'Rock', 'Reggae', 'Lo-fi', 'Instrumental', 'Soundtrack']  # Example list of genres
filter_and_update_csv(input_file, output_file, genres)


